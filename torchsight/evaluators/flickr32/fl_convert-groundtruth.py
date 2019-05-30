# -*- coding: utf-8 -*-
"""
 Helper script for the FlickrLogos-32 dataset.
 See http://www.multimedia-computing.de/flickrlogos/ for details.

 Please cite the following paper in your work:
 Scalable Logo Recognition in Real-World Images
 Stefan Romberg, Lluis Garcia Pueyo, Rainer Lienhart, Roelof van Zwol
 ACM International Conference on Multimedia Retrieval 2011 (ICMR11), Trento, April 2011.

 Author:   Stefan Romberg, stefan.romberg@informatik.uni-augsburg.de

 Notes:
  - Script was developed/tested on Windows with Python 2.7

 $Date: 2013-11-18 11:15:33 +0100 (Mo, 18 Nov 2013) $
 $Rev: 7621 $$Date: 2013-11-18 11:15:33 +0100 (Mo, 18 Nov 2013) $
 $HeadURL: https://137.250.173.47:8443/svn/romberg/trunk/romberg/research/FlickrLogos-32_SDK/FlickrLogos-32_SDK-1.0.4/scripts/fl_convert-groundtruth.py $
 $Id: fl_convert-groundtruth.py 7621 2013-11-18 10:15:33Z romberg $
"""
__version__ = "$Id: fl_convert-groundtruth.py 7621 2013-11-18 10:15:33Z romberg $"
__author__  = "Stefan Romberg, stefan.romberg@informatik.uni-augsburg.de"

# python built-in modules
import sys
from os.path import exists, basename, split, isdir
from collections import defaultdict

from flickrlogos import fl_read_groundtruth

#==============================================================================
#
#==============================================================================

def filename(x):
    """Returns the file name without the directory part including extension."""
    return split(x)[1]

def fl_read_cvt_groundtruth(flickrlogos_dir, out_file):
    #==========================================================================
    # check input: --flickrlogos
    #==========================================================================
    if flickrlogos_dir == "":
        print("ERROR: fl_read_cvt_groundtruth(): Missing ground truth directory (Missing argument --flickrlogos).")
        exit(1)

    if not exists(flickrlogos_dir):
        print("ERROR: fl_read_cvt_groundtruth(): Directory given by --flickrlogos does not exist: '"+str(flickrlogos_dir)+"'")
        exit(1)
        
    if not flickrlogos_dir.endswith('/') and not flickrlogos_dir.endswith('\\'):
        flickrlogos_dir += '/'

    gt_all = flickrlogos_dir + "all.txt"    
    if not exists(gt_all):
        print("ERROR: fl_read_cvt_groundtruth(): Ground truth file does not exist: '"+
              str(gt_all)+"' \nWrong directory given by --flickrlogos?")
        exit(1)

    #==========================================================================
    # read and process ground truth
    #==========================================================================
    gt_all, class_names = fl_read_groundtruth(gt_all)
    assert len(class_names) == 33   # 32 logo classes + "no-logo"
    assert len(gt_all) == 8240  # 32*10 (training set) + 32*30 (validation set)
    numImagesAll = len(gt_all)
    
    gt_images_per_class = defaultdict(set)
    for (im_file, logoclass) in gt_all.items():
        gt_images_per_class[logoclass].add( basename(im_file) )

    with open(out_file, "w") as f:
        #======================================================================
        # now loop over all items
        #======================================================================
        for no, (query_file, logoclass) in enumerate(gt_all.items()):
            print(no, query_file, logoclass)
            f.write(query_file+'\t'+logoclass+'\n')

#==============================================================================
if __name__ == '__main__': # MAIN
#============================================================================== 
    print("fl_convert-groundtruth.py\n"+__version__)

    # ----------------------------------------------------------------
    from optparse import OptionParser
    usage = "Usage: %prog --flickrlogos=<dataset root dir> --output=<output directory> "
    parser = OptionParser(usage=usage)

    parser.add_option("--flickrlogos", dest="flickrlogos", type=str, default=None,
                      help="Root directory of the FlickrLogos-32 dataset.\n")
    parser.add_option("-o","--output", dest="output", type=str, default='./all_gt.txt',
                      help="Optional: Output file, may be '-' for stdout. Default: stdout \n""")
    (options, args) = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        exit(1)

    #==========================================================================
    # show passed args
    #==========================================================================
    print("-"*79)
    print("ARGS:")
    print("FlickrLogos root dir (--flickrlogos):")
    print("  > '"+options.flickrlogos+"'")
    print("Output file ( --output):")
    print("  > '"+options.output+"'")
    print("-"*79)

    if options.flickrlogos is None:
        print("Missing argument: --flickrlogos=<FlickrLogos-32 root directory>. Exit.")
        exit(1)

    #==========================================================================
    # perform operation
    #==========================================================================
    fl_read_cvt_groundtruth(options.flickrlogos, options.output)

    print("Done.")
