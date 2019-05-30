# -*- coding: utf-8 -*-
"""
 Plots some charts for a given object recognition result.

 Tools for the FlickrLogos-32 dataset.
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
 $HeadURL: https://137.250.173.47:8443/svn/romberg/trunk/romberg/research/FlickrLogos-32_SDK/FlickrLogos-32_SDK-1.0.4/scripts/fl_plot_classification_results.py $
 $Id: fl_plot_classification_results.py 7621 2013-11-18 10:15:33Z romberg $
"""
__version__ = "$Id: fl_plot_classification_results.py 7621 2013-11-18 10:15:33Z romberg $"
__author__  = "Stefan Romberg, stefan.romberg@informatik.uni-augsburg.de"

import sys, os.path
from os.path import exists, basename
from collections import defaultdict

import matplotlib.pyplot as plt

from flickrlogos import fl_read_groundtruth, fl_read_csv
from flickrlogos.ConfusionMatrix import ConfusionMatrix

#==============================================================================
#
#==============================================================================

def plot_bar_chart(x_names, y_values, title="", xlabel="", ylabel="", y_perc=None, figsize=None):
    """Plots bar char."""
    assert isinstance(x_names, list)
    assert isinstance(y_values, list)
    
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111)
    
    x = range(0, len(x_names))
    x = [ xx + 0.5 for xx in x ]
    ax.bar(x, y_values)

    ax.grid(True)
    ax.set_autoscale_on(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(x)+1.25 )
    ax.set_ylim(0, max(y_values)+0.25 )

    plt.xticks( [xx + 0.5 for xx in x], x_names, rotation=50, horizontalalignment='right')
    
    for i,y_val in enumerate(y_values):
        plt.text(0.65+i*1.000, 0.5, str(y_val).rjust(2), color="white", fontsize="small")
    
    if y_perc is not None:
        for i,y_val in enumerate(y_perc):
            s = str(int(round(y_val)))
            plt.text(0.60+i*1.000, 2.0, s.rjust(3), color="white", fontsize="x-small")
            plt.text(0.60+i*1.000, 2.8, "  %",      color="white", fontsize="x-small")
                    
    plt.subplots_adjust(top=0.93,
                        bottom=0.18,
                        left=0.08,
                        right=0.97)
    plt.tight_layout(pad=1.0)
    return plt

#==============================================================================
# main()
#==============================================================================
def main(flickrlogos_dir, result_file, output_prefix,
         output_formats="_.png,.pdf,.eps,.svg", #  PNG files end on '_.png', this makes latex ignore them 
         show_plots=False):
    """main()"""

    graphic_formats = []
    if output_formats is not None and output_formats != None:
        graphic_formats = output_formats.split(",")

    if not exists(flickrlogos_dir):
        print("ERROR: main(): Directory does not exist: '"+str(flickrlogos_dir)+"'")
        exit(1)

    if not flickrlogos_dir.endswith('/') and not flickrlogos_dir.endswith('\\'):
        flickrlogos_dir += '/'

    gt_all = flickrlogos_dir + "all.txt"

    if not exists(gt_all):
        print("ERROR: File does not exist: '"+str(gt_all)+"'")
        exit(1)

    if not exists(result_file):
        print("ERROR: File does not exist: '"+str(result_file)+"'")
        exit(1)

    #==========================================================================
    # load data
    #==========================================================================
    groundtruth, class_names = fl_read_groundtruth(gt_all)

    actual_data = fl_read_csv(result_file)
    actual_data = [ (image_id, classification.lower(), confidence) for image_id, classification, confidence in actual_data ]
        
    for image_id, classification, confidence in actual_data:
        assert classification in class_names, (classification, class_names)

    #==========================================================================
    # Process data, count TPs for each class
    #==========================================================================
    tp_per_class      = defaultdict(int)
    for image_id, classification, confidence in actual_data:
        im = image_id + ".jpg"
        if not im in groundtruth:
            continue
        
        if groundtruth[im] == classification:
            tp_per_class[classification] += 1
    
    # remove no-logo class from bar chart    
    class_names_logosonly = [ c for c in class_names if c != "no-logo" ]
    
    # fetch TP counts
    tp_values = [ tp_per_class[c] for c in class_names_logosonly ]
    tp_perc   = [ (float(tp_per_class[c])/30.0)*100.0 for c in class_names_logosonly ]
    
    # prepare class names for visualization with proper capitalization and abbreviations
    class_names_logosonly = [ c.capitalize() if len(c) > 3 else c.upper() for c in class_names_logosonly ]

    #==========================================================================
    # plot
    #==========================================================================
    p = plot_bar_chart(x_names=class_names_logosonly,
                       y_values=tp_values,
                       y_perc=tp_perc,
                       title="True Positives per Class",
                       xlabel="Class",
                       ylabel="True Positives",
                       figsize=(10, 6))

    filename = output_prefix + "results-tp-per-class"
    for format_ext in graphic_formats:
        p.savefig(filename+format_ext, dpi=300, transparent=True)

    if show_plots:
        p.show()

    #==========================================================================
    # plot confusion matrix
    #==========================================================================
    cm = ConfusionMatrix()
    for image_id, classification, confidence in actual_data:
        im = image_id + ".jpg"
        if not im in groundtruth:
            continue
        cm.add_result(groundtruth[im], classification)

    p = cm.plot()

    filename = output_prefix + "results-conf-mat"
    for format_ext in graphic_formats:
        p.savefig(filename+format_ext, dpi=300, transparent=True)

    if show_plots:
        p.show()

#==============================================================================
if __name__ == '__main__': # MAIN
#============================================================================== 
    print("fl_plot_classification_results.py\n"+__version__)

    # ----------------------------------------------------------------
    from optparse import OptionParser
    usage = "Usage: %prog --flickrlogos=<dataset root dir> --classification=<classification file> "
    parser = OptionParser(usage=usage)

    parser.add_option("--flickrlogos", dest="flickrlogos", type=str, default="",
                      help="Base (root) directory of the FlickrLogos-32 dataset\n")
    parser.add_option("--classification", dest="classification", type=str, default="",
                      help="""File containing images and the corresponding detected classes " +
                        "in the following format: <image id>, <detected class name>, "+
                        "<confidence value or 1 if class was detected, 0 otherwise> \n""")
    parser.add_option("--output-prefix", dest="output_prefix", type=str, default="",
                      help="""Output directory for generated plots.""")
    parser.add_option("--output-formats", dest="output_formats", type=str, default="_.png,.pdf,.eps,.svg",
                      help="""Desired output formats for plots\nDefault: '_.png,.pdf,.eps,.svg'.\n""")
    parser.add_option("--show-images", dest="show_images", default="yes",
                      help="""Show plots to user.\n""")

    (options, args) = parser.parse_args()

    if len(sys.argv) < 3:
        parser.print_help()
        exit(1)

    if options.show_images.lower() == "false" or options.show_images.lower() == "no":
        options.show_images = False

    #==========================================================================
    # show passed args
    #==========================================================================
    print("-"*79)
    print(" --flickrlogos base dir: '"+str(options.flickrlogos)+"'")
    print(" --classification:       '"+str(options.classification)+"'")
    print(" --output-prefix:        '"+str(options.output_prefix)+"'")
    print(" --output-formats:       '"+str(options.output_formats)+"'")
    print(" --show-images:          '"+str(options.show_images)+"'")
    print("-"*79)
    print("")

    if not exists(options.flickrlogos):
        print("ERROR: Invalid argument --flickrlogos: Directory does not exist: '"+str(options.flickrlogos)+"'")
        exit(1)

    if not exists(options.classification):
        print("ERROR: Invalid argument --classification: File does not exist: '"+str(options.classification)+"'")
        exit(1)

    #==========================================================================
    #
    #==========================================================================

    main(options.flickrlogos,
         options.classification,
         options.output_prefix,
         options.output_formats,
         options.show_images)

    print("Done")
