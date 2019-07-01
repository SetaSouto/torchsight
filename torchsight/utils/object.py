"""A module with utilities to work with json objects."""
import json

from .merge import merge_dicts


class JsonObject():
    """A class that allows to transform a dict into an object to make
    less verbose access to the properties of the dict.

    Is very useful when you are working with very nested dicts or if you want to
    deeply merge two dicts and keep working.

    If your are using a linter like pylint you can found some errors
    in your code telling that "Instance of '<this class>' has no '<attribute' member."
    You can ignore this class adding to your .pylintrc the following text:
    [TYPECHECK]
    ignored-classes=JsonObject,OtherClass,AnotherClass
    """

    def __init__(self, data):
        """Initialize the object.

        Arguments:
            data (dict or str): The dict to be transformed to object or a path
                to a json file.
        """
        if isinstance(data, str):
            with open(data, 'r') as file:
                data = json.loads(file.read())

        for key, value in data.items():
            setattr(self, key, self._transform(value))

    def _transform(self, value):
        """Transform the given value to set it as attribute.

        If the value has more values inside (like a list or another dict) it transforms it
        recursive.

        Arguments:
            value: The value to transform recursively or not.

        Returns:
            The transformed value.
        """
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._transform(v) for v in value])

        if isinstance(value, dict):
            return self.__class__(value)

        return value

    def dict(self):
        """Get the original dict based on the __dict__ of this instance.

        Returns:
            dict: The original dict of this instance.
        """
        original = {}

        for key, value in self.__dict__.items():
            original[key] = value.dict() if isinstance(value, self.__class__) else value

        return original

    def __str__(self):
        """The human readable version of the object.

        Returns:
            str: The string representing this instance.
        """
        return json.dumps(self.dict(), indent=2)

    def merge(self, data, verbose=False):
        """Deep merge the current object with another dict or JsonObject instance.

        If the actual data collides with the new data, the new data take precedence, recursively.

        Arguments:
            data (dict or JsonObject): The new data to merge with the actual data.
            verbose (bool, optional): If True it will print a message when adding/updating values.

        Returns:
            JsonObject: The self instance.
        """
        if data is None:
            return self

        if isinstance(data, self.__class__):
            data = data.dict()

        data = merge_dicts(self.dict(), data, verbose=verbose)

        self.__class__.__init__(self, data)

        return self

    def keys(self):
        """Get the keys of the instance.

        Returns:
            dict_keys: The list with the keys of this instance.
        """
        return self.__dict__.keys()

    def __getitem__(self, key):
        """Get an item of the object by key.

        This method allows to make the object subscriptable and make things like **object.

        Arguments:
            key: The key of the value to get.

        Returns:
            The value for the given key.
        """
        return getattr(self, key)
