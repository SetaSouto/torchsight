"""Anchors module"""
import torch
from torch import nn

from ..metrics import iou as compute_iou


class Anchors(nn.Module):
    """Module to generate anchors for a given image.

    Anchors could be generated for different sizes, scales and ratios.

    Anchors are useful for object detectors, so, for each location (or "pixel") of the feature map
    it regresses a bounding box for each anchor and predict the class for each bounding box.
    So, for example, if we have a feature map of (10, 10, 2048) and 9 anchors per location we produce
    10 * 10 * 9 bounding boxes.

    The bounding boxes has shape (4) for the x1, y1 (top left corner) and x2, y2 (bottom right corner).
    Also each bounding box has a vector with the probabilities for the C classes.

    In this case we could generate different anchors depending on 3 basic variables:
    - Size: Base size of the anchor.
    - Scale: Scale the base size to different scales.
    - Aspect ratio: Disfigure the anchor to different aspect ratios.

    And then shift (move) the anchors according to the image. So each location of the feature map has
    len(scales) * len(ratios) = A anchors.
    As the feature maps has different strides depending on the network, we need the strides values to adjust
    the shift of each anchor box.
    For example, if we have an image with 320 * 320 pixels and apply a CNN with stride 10 we have a feature map
    with 32 * 32 locations (320 / 10). So, each location represents a region of 10 * 10 pixels so to center the
    anchor to the center of the location we need to move it 5 pixels to the right and 5 pixels down so the center
    of the anchor is at the center of the location. With this, we move the anchor to the (i, j) position of
    the feature map by moving it by (i * stride, j * stride) pixels and then center it by moving it by
    (stride // 2, stride // 2), so the final movement is (i * stride + stride // 2, j * stride + stride // 2).

    --- Feature Pyramid Network ---

    The anchors could be for a Feature Pyramid Network, so its predicts for several feature maps
    of different scales. This is useful to improve the precision over different object scales (very little ones
    and very large ones).

    The feature pyramid network (FPN) produces feature maps at different scales, so we use different anchors per scale,
    for example in the original paper of RetinaNet they use images of size 600 * 600 and the 5 levels of the FPN
    (P3, ..., P7) with anchors with areas of 32 * 32 to 512 * 512.

    Each anchor size is adapted to three different aspect ratios {1:2, 1:1, 2:1} and to three different scales
    {2 ** 0, 2 ** (1/3), 2 ** (2/3)} according to the paper.

    So finally, we have 3 * 3 = 9 anchors based on one size, totally 3 * 3 * 5 = 45 different anchors for the total
    network.
    Keep in mind that only 3 * 3 anchors are used per location in the feature map and that the FPN produces 5 feature
    maps, that's why we have 3 * 3 * 5 anchors.

    Example:
    If we have anchors_sizes = [32, 64, 128, 256, 512] then anchors with side 32 pixels are used for the P3 output of
    the FPN, 64 for the P4, ... , 512 for the P7.
    Using the scales {2 ** 0, 2 ** (1/3), 2 ** (2/3)} we can get anchors from 32 pixels of side to 813 pixels of side.
    """

    def __init__(self,
                 sizes=[32, 64, 128, 256, 512],
                 scales=[2 ** 0, 2 ** (1/3), 2 ** (2/3)],
                 ratios=[0.5, 1, 2],
                 strides=[8, 16, 32, 64, 128],
                 device=None):
        """Initialize the network.

        The base values are from the original paper of RetinaNet.

        Args:
            sizes (sequence, optional): The sizes of the base anchors that will be scaled and deformed for each
                feature map expected.
            scales (sequence, optional): The scales to modify each base anchor.
            ratios (sequence, optional): The aspect ratios to deform the scaled base anchors.
            strides (sequence, optional): The stride applied to the image to generate the feature map
                for each base anchor size.
            device (str): The device where to run the computations.
        """
        super(Anchors, self).__init__()

        if len(sizes) != len(strides):
            # As we have one size for each feature map we need to know the stride applied to the image
            # to get that feature map
            raise ValueError('"sizes" and "strides" must have the same length')

        self.strides = strides
        self.n_anchors = len(scales) * len(ratios)
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.base_anchors = self.generate_anchors(sizes, scales, ratios)

    def to(self, device):
        """Move the module to the given device.

        Arguments:
            device (str): The device where to move the module.
        """
        self.device = device
        self.base_anchors = self.base_anchors.to(device)
        return super(Anchors, self).to(device)

    def forward(self, images):
        """Generate anchors for the given image.

        If we have multiplies sizes and strides we assume that we'll have different feature maps at different
        scales. This method returns all the anchors together for all the feature maps in order of the sizes.

        It returns a Tensor with shape:
        (batch size, total anchors, 4)

        Why 4?
        The four parameters x1, y1, x2, y2 for each anchor.

        How many is total anchors?
        We have len(sizes) * len(strides) = A anchors per location of each feature map.
        We have len(sizes) = len(strides) = F different feature maps based on the image (for different scales).
        For each 'i' feature map we have (image width / strides[i]) * (image height / strides[i]) = L locations.
        So total anchors = A * F * L.

        So the anchors, for each image, are in the form of:

        [[x1, y1, x2, y2],
         [x1, y1, x2, y2],
         ...]

        And they are grouped by:
        - Every A anchors we have a location (x, y) of a feature map
        - The next A anchors are for the (x+1, y) of the feature map.
        - After all the columns (x) of the feature map then the next A anchors are for the (x, y + 1) location
        - After all rows and all columns we pass to the next feature map, i.e., the next size of anchors.

        So, the thing to keep in mind is that the regression values must follow the same order, first A values
        for the location (0, 0) of the feature map of smallest size, then the next A values for the (1, 0)
        location, until all the columns are covered, then increase the rows (the next A values must be for the
        location (0, 1), the next A for the (1, 1), and so on).

        Args:
            image (torch.Tensor): The image from we'll get the feature maps and generate the anchors for.

        Returns:
            torch.Tensor: A tensor with shape (batch size, total anchors, 4).
        """
        # We have the base anchors that we need to shift and adjust to the given image.
        # The base anchors have shape (len(sizes), len(scales) * len(ratios), 4)
        # For a given stride, say i, we need to generate (image width / stride) * (image height / stride)
        # anchors for each one of the self.base_anchors[i, :, :]
        anchors = []
        for image in images:
            image_anchors = []
            for index, stride in enumerate(self.strides):
                base_anchors = self.base_anchors[index, :, :]  # Shape (n_anchors, 4)
                height, width = image.shape[1:]
                # Get dimensions of the feature map
                feature_height, feature_width = round(height / stride), round(width / stride)
                # We need to move each anchor (i * shift, j * shift) for each (i,j) location in the feature map
                # to center the anchor on the center of the location
                shift = stride * 0.5
                # We need a tensor with shape (feature_height, feature_width, 4)
                # shift_x shape: (feature_height, feature_width, 1) where each location has the index of the
                # x position of the location in the feature map plus the shift value
                shift_x = torch.arange(feature_width).to(self.device).type(torch.float)
                shift_x = stride * shift_x.unsqueeze(0).unsqueeze(2).repeat(feature_height, 1, 1) + shift
                # shift_y shape: (feature_height, feature_width, 1) where each location has the index of the
                # y position of the location in the feature map plus the shift value
                shift_y = torch.arange(feature_height).to(self.device).type(torch.float)
                shift_y = stride * shift_y.unsqueeze(1).unsqueeze(2).repeat(1, feature_width, 1) + shift
                # The final shift will have shape (feature_height, feature_width, 4 * n_anchors)
                shift = torch.cat([shift_x, shift_y, shift_x, shift_y], dim=2).repeat(1, 1, self.n_anchors)
                # As pytorch interpret the channels in the first dimension it could be better to have a shift
                # with shape (4 * anchors, feature_height, feature_width) like an image (channels, height, width)
                shift = shift.permute((2, 0, 1))
                base_anchors = base_anchors.view(self.n_anchors * 4).unsqueeze(1).unsqueeze(2)
                # As shift has shape (4* n_anchors, feature_height, feature_width) and base_anchors has shape
                # (4 * n_anchors, 1, 1) this last one broadcast to the shape of shift
                final_anchors = shift + base_anchors
                # As the different strides generate different heights and widths for each feature map we cannot
                # stack the anchors for each feature map of the image, but we can concatenate the anchors if we
                # reshape them to (4, n_anchors * feature_height * feature_width) but to have the parameters in the
                # last dimension we can permute the dimensions.
                # Until now we have in the channels the values for each x1, y1, x2, y2, but the view method
                # follow the order of rows, cols, channels, so if we want to keep the values per each anchor
                # continuous we need to apply the permutation first and then the view.
                # Finally we get shape (total anchors, 4)
                final_anchors = final_anchors.permute((1, 2, 0)).contiguous().view(-1, 4)
                image_anchors.append(final_anchors)
            # Now we have an array with the anchors with shape (n_anchors * different heights * widths, 4) so we can
            # concatenate by the first dimension to have a list with anchors (with its 4 parameters) ordered by
            # sizes, x_i, y_i, all combination of aspect ratios vs scales
            # where x_i, y_i are the positions in the feature map for the size i
            anchors.append(torch.cat(image_anchors, dim=0))
        # Now as each image has its anchors with shape
        # (n_anchors * feature_height_i * feature_width_i for i in range(len(strides)), 4)
        # we can stack the anchors for each image and generate a tensor with shape
        # (batch size, n_anchors * feature_height_i * feature_width_i for i in range(len(strides)), 4)
        return torch.stack(anchors, dim=0)

    def generate_anchors(self, sizes, scales, ratios):
        """Given a sequence of side sizes generate len(scales) *  len(ratios) = n_anchors anchors per size.
        This allow to generate, for a given size, all the possibles combinations between different aspect
        ratios and scales.

        To get the total anchors for a given size, say 0, you could do:
        ```
        anchors = self.generate_anchors(sizes, scales, ratios)
        anchors[0, :, :]
        ```
        Or to get the 4 parameters for the j anchor for the i size:
        ```
        anchors[i, j, :]
        ```

        Args:
            anchors_sizes (sequence): Sequence of int that are the different sizes of the anchors.

        Returns:
            torch.Tensor: Tensor with shape (len(anchors_sizes), len(scales) * len(ratios), 4).
        """
        # First we are going to compute the anchors as center_x, center_y, height, width
        anchors = torch.zeros((len(sizes), self.n_anchors, 4), dtype=torch.float).to(self.device)
        sizes, scales, ratios = [torch.Tensor(x).to(self.device) for x in [sizes, scales, ratios]]
        # Start with height = width = 1
        anchors[:, :, 2:] = torch.Tensor([1., 1.]).to(self.device)
        # Scale each anchor to the correspondent size. We use unsqueeze to get sizes with shape (len(sizes), 1, 1)
        # and broadcast to (len(sizes), n_anchors, 4)
        anchors *= sizes.unsqueeze(1).unsqueeze(1)
        # Multiply the height for the aspect ratio. We repeat the ratios len(scales) times to get all the aspect
        # ratios for each scale. We unsqueeze the ratios to get the shape (1, n_anchors) to broadcast to
        # (len(sizes), n_anchors)
        anchors[:, :, 2] *= ratios.repeat(len(scales)).unsqueeze(0)
        # Adjust width and height to match the area size * size
        areas = sizes * sizes  # Shape (len(sizes))
        height, width = anchors[:, :, 2], anchors[:, :, 3]  # Shapes (len(sizes), n_anchors)
        adjustment = torch.sqrt((height * width) / areas.unsqueeze(1))
        anchors[:, :, 2] /= adjustment
        anchors[:, :, 3] /= adjustment
        # Multiply the height and width by the correspondent scale. We repeat the scale len(ratios) times to get
        # one scale for each aspect ratio. So scales has shape (1, n_anchors, 1) and
        # broadcast to (len(sizes), n_anchors, 2) to scale the height and width.
        anchors[:, :, 2:] *= scales.unsqueeze(1).repeat((1, len(ratios))).view(1, -1, 1)

        # Return the anchors but not centered nor with height or width, instead use x1, y1, x2, y2
        height, width = anchors[:, :, 2].clone(), anchors[:, :, 3].clone()
        center_x, center_y = anchors[:, :, 0].clone(), anchors[:, :, 1].clone()

        anchors[:, :, 0] = center_x - (width * 0.5)
        anchors[:, :, 1] = center_y - (height * 0.5)
        anchors[:, :, 2] = center_x + (width * 0.5)
        anchors[:, :, 3] = center_y + (height * 0.5)

        return anchors

    @staticmethod
    def transform(anchors, deltas):
        """Adjust the anchors with the regression values (deltas) to obtain the final bounding boxes.

        It uses the standard box parametrization from R-CNN:
        https://arxiv.org/pdf/1311.2524.pdf (Appendix C)

        Args:
            anchors (torch.Tensor): Anchors generated for each image in the batch.
                Shape:
                    (batch size, total anchors, 4)
            deltas (torch.Tensor): The regression values to adjust the anchors to generate the real
                bounding boxes as x1, y2, x2, y2 (top left corner ahd bottom right corner).
                Shape:
                    (batch size, total anchors, 4)

        Returns:
            torch.Tensor: The bounding boxes with shape (batch size, total anchors, 4).
        """
        widths = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        center_x = anchors[:, :, 0] + (widths / 2)
        center_y = anchors[:, :, 1] + (heights / 2)

        delta_x = deltas[:, :, 0]
        delta_y = deltas[:, :, 1]
        delta_w = deltas[:, :, 2]
        delta_h = deltas[:, :, 3]

        predicted_x = (delta_x * widths) + center_x
        predicted_y = (delta_y * heights) + center_y
        predicted_w = torch.exp(delta_w) * widths
        predicted_h = torch.exp(delta_h) * heights

        # Transform to x1, y1, x2, y2
        predicted_x1 = predicted_x - (predicted_w / 2)
        predicted_y1 = predicted_y - (predicted_h / 2)
        predicted_x2 = predicted_x + (predicted_w / 2)
        predicted_y2 = predicted_y + (predicted_h / 2)

        return torch.stack([
            predicted_x1,
            predicted_y1,
            predicted_x2,
            predicted_y2
        ], dim=2)

    @staticmethod
    def clip(batch, boxes):
        """Given the boxes predicted for the batch, clip the boxes to fit the width and height.

        This means that if the box has any side outside the dimensions of the images the side is adjusted
        to fit inside the image. For example, if the image has width 800 and the right side of a bounding
        box is at x = 830, then the right side will be x = 800.

        Args:
            batch (torch.Tensor): The batch with the images. Useful to get the width and height of the image.
                Shape:
                    (batch size, channels, width, height)
            boxes (torch.Tensor): A tensor with the parameters for each bounding box.
                Shape:
                    (batch size, number of bounding boxes, 4).
                Parameters:
                    boxes[:, :, 0]: x1. Location of the left side of the box.
                    boxes[:, :, 1]: y1. Location of the top side of the box.
                    boxes[:, :, 2]: x2. Location of the right side of the box.
                    boxes[:, :, 3]: y2. Location of the bottom side of the box.

        Returns:
            torch.Tensor: The clipped bounding boxes with the same shape.
        """
        _, _, height, width = batch.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes

    @staticmethod
    def assign(anchors, annotations, thresholds=None):
        """Assign the correspondent annotation to each anchor.

        We know that in a feature map we'll have A anchors per location (i.e. feature map's height * width * A total
        anchors), where these A anchors have different sizes and aspect ratios.
        Each one of this anchors could have an 'assigned annotation', that is the annotation that has bigger
        Intersection over Union (IoU) with the anchor.

        So, all the anchors could have an assigned annotation, but, what we do with the anchors that are containing
        background and none object? That's the reason that we must provide some thresholds: one to keep only the
        anchors that are containing objects and one threshold to decide if the anchor is background or not.
        The anchors that has IoU between those threshold are ignored.

        With this we can train anchors to fit the annotation and others to fit the background class. And obviously
        keep a centralized way to handle this behavior.

        Arguments:
            anchors (torch.Tensor): The base anchors. They must have 4 values: x1, y1 (top left corner), x2, y2 (bottom
                right corner).
                Shape:
                    (number of anchors, 4)
            annotations (torch.Tensor): The real ground truth annotations of the image. They must have at least the
                same 4 values.
                Shape:
                    (number of annotations, 4+)
            thresholds (dict): A dict with the 'object' (float) threshold and the 'background' threshold.
                If the IoU between an anchor and an annotation is bigger than 'object' threshold it's selected as
                an object anchor, if the IoU is below the 'background' threshold is selected as a 'background' anchor.

        Returns:
            torch.Tensor: The assigned annotations. The annotation at index i is the annotation associated to the
                anchor at index i. So for example, if we have the anchors tensor and we want to get the annotation to
                the i-th anchor we could simply take this value returned and the get i-th element too. Example:
                >>> assigned_annotations, *_ = Anchors.assign(anchors, annotations)
                >>> assigned_annotations[10]  # The annotation assigned to the 10th anchor.
                Shape:
                    (number of anchors, 4+)
            torch.Tensor: A mask that indicates which anchors are selected to be objects.
            torch.Tensor: A mask that indicates which anchors are selected to be background.
                Keep in mind that if the thresholds are not the same some anchors could not be objects nor background.
            torch.Tensor: The IoU of the selected anchors as objects. Obviously all of them are over the 'object'
                threshold.
        """
        if annotations is None or annotations.shape[0] == 0:
            # There are no assigned annotations
            num_anchors = anchors.shape[0]
            assigned_annotations = -1 * anchors.new_ones(num_anchors, 5)
            # There are no selected anchors as objects
            objects_mask = anchors.new_zeros(num_anchors).byte()
            # All the anchors are background
            background_mask = anchors.new_ones(num_anchors).byte()
            # There are no iou between anchors and objects
            iou_objects = anchors.new_zeros(0)
            return assigned_annotations, objects_mask, background_mask, iou_objects

        default_thresholds = {'object': 0.5, 'background': 0.4}

        if thresholds is None:
            thresholds = default_thresholds
        if 'object' not in thresholds:
            thresholds['object'] = default_thresholds['object']
        if 'background' not in thresholds:
            thresholds['background'] = default_thresholds['background']

        iou = compute_iou(anchors, annotations)  # (number of anchors, number of annotations)
        iou_max, iou_argmax = iou.max(dim=1)  # (number of anchors)
        # Each anchor is associated to a bounding box. Which one? The one that has bigger iou with the anchor
        assigned_annotations = annotations[iou_argmax, :]  # (number of anchors, 4+)
        # Only train bounding boxes where its base anchor has an iou with an annotation over iou_object threshold
        objects_mask = iou_max > thresholds['object']
        iou_objects = iou_max[objects_mask]
        background_mask = iou_max < thresholds['background']

        return assigned_annotations, objects_mask, background_mask, iou_objects
