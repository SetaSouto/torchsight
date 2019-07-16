# -*- coding: utf-8 -*-
"""
 Evaluation script for the FlickrLogos-32 dataset.
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
 $HeadURL: https://137.250.173.47:8443/svn/romberg/trunk/romberg/research/FlickrLogos-32_SDK/FlickrLogos-32_SDK-1.0.4/scripts/fl_eval_retrieval.py $
 $Id: fl_eval_retrieval.py 7621 2013-11-18 10:15:33Z romberg $
"""
__version__ = "$Id: fl_eval_retrieval.py 7621 2013-11-18 10:15:33Z romberg $"
__author__  = "Stefan Romberg, stefan.romberg@informatik.uni-augsburg.de"

# python built-in modules
import sys
from os.path import exists, basename, split, isdir
from collections import defaultdict

from .flickrlogos import fl_read_groundtruth, fl_read_csv, fl_ap, fl_mean, fl_sdev, Tee

#==============================================================================
#
#==============================================================================

def filename(x):
    """Returns the filename without the directory part including extension."""
    return split(x)[1]

def sround(x, arg):
    if isinstance(x, float):
        return str(round(x, arg))
    else:
        return str(x)

def fl_read_retrieval_results_format2(result_file):
    """
    Reads the retrieval results from an ASCII file.

    Format:
    1st column:     Path to image file, should not contain spaces
    2nd column:     Score/similarity, higher scores are better.
    Other columns:  Ignored if present.
    The first line contains the query image with an arbitrary score (i.e. 1.0).
    """
    lines = fl_read_csv(result_file, delimiters=" \t,;")

    # remove first line, contains header = query image
    lines = lines[1:]

    if len(lines) > 0:
        assert len(lines[0]) >= 2

    # Returns a list: [ (score, file0, ), (score, file1), ... ]
    return [ (float(x[1]), x[0]) for x in lines ]

def fl_process_retrieval_results(results, sort, T_score):

    # sort results
    if sort == "sort-desc":
        results = list(reversed(sorted(results)))
    elif sort == "sort-asc":
        results = sorted(results)
    else:
        pass

    # keep all files with score > T_score
    results = [ (score, imfile) for (score, imfile) in results if score > T_score ]

    return results

#==============================================================================
#
#==============================================================================

def fl_eval_retrieval(indir, flickrlogos_dir, verbose):
    #==========================================================================
    # check input: --flickrlogos
    #==========================================================================
    if flickrlogos_dir == "":
        print("ERROR: fl_eval_retrieval(): Missing ground truth directory (Missing argument --flickrlogos).")
        exit(1)

    if not exists(flickrlogos_dir):
        print("ERROR: fl_eval_retrieval(): Directory given by --flickrlogos does not exist: '"+str(flickrlogos_dir)+"'")
        exit(1)

    if not flickrlogos_dir.endswith('/') and not flickrlogos_dir.endswith('\\'):
        flickrlogos_dir += '/'

    gt_trainvalset        = flickrlogos_dir + "trainvalset.txt"
    gt_testset_logosonly  = flickrlogos_dir + "testset-logosonly.txt"

    if not exists(gt_trainvalset):
        print("ERROR: fl_eval_retrieval(): Ground truth file does not exist: '"+
              str(gt_trainvalset)+"' \nWrong directory given by --flickrlogos?")
        exit(1)

    if not exists(gt_testset_logosonly):
        print("ERROR: fl_eval_retrieval(): Ground truth file does not exist: '"+
              str(gt_testset_logosonly)+"'\nWrong directory given by --flickrlogos?")
        exit(1)

    #==========================================================================
    # check input: --indir
    #==========================================================================
    if indir.startswith("file:///"): # for copy-pasting stuff from browser into console
        indir = indir[8:]

    if not exists(indir):
        print("ERROR: fl_eval_retrieval(): Directory given by --indir does not exist: '"+str(indir)+"'")
        exit(1)

    #==========================================================================
    # read and process ground truth
    #==========================================================================

    gt_indexed, class_names = fl_read_groundtruth(gt_trainvalset)
    assert len(class_names) == 33   # 32 logo classes + "no-logo"
    assert len(gt_indexed) == 4280  # 32*10 (training set) + 32*30 (validation set)

    numImagesIndexed = len(gt_indexed)

    gt_queries, class_names = fl_read_groundtruth(gt_testset_logosonly)
    assert len(class_names) == 32   # 32 logo classes, logos only
    assert len(gt_queries) == 960   # 32*30 (test set)
    numQueries = len(gt_queries)

    gt_images_per_class = defaultdict(set)
    for (im_file, logoclass) in gt_indexed.items():
        gt_images_per_class[logoclass].add( basename(im_file) )

    #==========================================================================
    # now loop over all queries:
    #    - fetch result list/ranking
    #    - compute AP, P, R, etc..
    #==========================================================================
    APs      = []
    RRs      = []
    top4s    = []
    numEmpty = 0
    TP       = 0
    TP_FP    = 0
    TP_FN    = 0

    for no, (query_file, logoclass) in enumerate(gt_queries.items()):
        query       = basename(query_file)
        result_file = indir + "/" + query + ".result2.txt"

        # current class
        cur_class = gt_queries[query]
        #print("cur_class:" + cur_class)

        if not exists(result_file):
            print("ERROR: Missing result file: '"+str(result_file)+"'! Exit.\n")
            exit(1)

        #----------------------------------------------------------------------
        # - read retrieval results
        # - sort by descending score
        # - remove results with score <= T_score
        #----------------------------------------------------------------------
        results = fl_read_retrieval_results_format2(result_file)

        T_score = 0
        results = fl_process_retrieval_results(results, "sort-desc", T_score)

        #----------------------------------------------------------------------
        # save rank of each item
        #----------------------------------------------------------------------
        ranked_list = []
        for (score, im_file) in results:
            ranked_list.append( filename(im_file) )

        if len(ranked_list) == 0:
            numEmpty += 1

        #----------------------------------------------------------------------
        # compute mAP
        #----------------------------------------------------------------------
        pos  = gt_images_per_class[cur_class]
        assert len(pos) == 40
        amb  = set() # empty set, no images are ignored
        AP   = fl_ap(pos, amb, ranked_list)
        APs.append(AP)

        #----------------------------------------------------------------------
        # compute precision + recall, count TP, FP, FN for now
        #----------------------------------------------------------------------
        tp_set = set(ranked_list).intersection(pos)

        P = 0.0
        if len(ranked_list) != 0:
            P = float(len(tp_set)) / float(len(ranked_list))

        R = 0.0
        if len(pos) != 0:
            R =  float(len(tp_set)) / float(len(pos))

        TP    += len(tp_set)
        TP_FP += len(ranked_list)
        TP_FN += len(pos)

        #----------------------------------------------------------------------
        # compute AverageTop4 score, i.e. P@4 * 4.0
        #----------------------------------------------------------------------
        count4 = 0
        for x in ranked_list[0:4]:
            if x in pos:
                count4 += 1

        top4s.append(count4)

        #----------------------------------------------------------------------
        # compute response ratio (RR)
        #----------------------------------------------------------------------
        RR = float(len(results)) / float(numImagesIndexed)
        RRs.append(RR)
        
        if verbose:
            sys.stderr.write("\r Processing retrieval result file "+str(no+1)+"/"+str(len(gt_queries))+" ...")

    if verbose:
        print(" Done")
    #==========================================================================
    # DONE
    #==========================================================================
    assert len(APs)   == numQueries, (len(APs), numQueries)
    assert len(top4s) == numQueries, (len(top4s), numQueries)
    assert len(RRs)   == numQueries, (len(RRs), numQueries)

    print("-"*79)
    print(" Results ")
    print("-"*79)
    print(" indir: '"+str(indir)+"'")    
    print("")
    print(" Number of queries:                 "+str(numQueries))
    print(" Number of empty result lists:      "+str(numEmpty))
    prec = 4

    mAP    = fl_mean(APs)
    mAP_sd = fl_sdev(APs)
    print(" ==> mean average precision (mAP):  "+sround(mAP, prec).ljust(8)+" (stddev: "+sround(mAP_sd, prec)+")")

    avgTop4    = fl_mean(top4s)
    avgTop4_sd = fl_sdev(top4s)
    print(" ==> Avg. top 4 score (4*P@4):      "+sround(avgTop4, prec).ljust(8)+" (stddev: "+sround(avgTop4_sd, prec)+")")

    mP    = float(TP) / float(TP_FP)
    print(" ==> mean precision (mP):           "+sround(mP, prec).ljust(8))

    mR    = float(TP) / float(TP_FN)
    print(" ==> mean recall (mR)::             "+sround(mR, prec).ljust(8))

    RR    = fl_mean(RRs)
    RR_sd = fl_sdev(RRs)
    print(" ==> response ratio (RR):           "+sround(RR, prec).ljust(8)+" (stddev: "+sround(RR_sd, prec)+")")

#==============================================================================
if __name__ == '__main__': # MAIN
#==============================================================================    
    from optparse import OptionParser
    usage = "Usage: %prog --flickrlogos=<dataset root dir> --indir=<directory with result files> "
    parser = OptionParser(usage=usage)

    parser.add_option("--flickrlogos", dest="flickrlogos", type=str, default=None,
                      help="Root directory of the FlickrLogos-32 dataset.\n")
    parser.add_option("--indir", dest="indir", type=str, default=None,
                      help="Directory holding the retrieval result files (*.results2.txt).")
    parser.add_option("-o","--output", dest="output", type=str, default="-",
                      help="Optional: Output file, may be '-' for stdout. Default: stdout \n""")
    parser.add_option("-v","--verbose", dest="verbose", action="store_true", default=False, 
                      help="Optional: Flag for verbose output. Default: False\n""")
    (options, args) = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        exit(1)

    #==========================================================================
    # show passed args
    #==========================================================================
    if options.verbose:
        print("fl_eval_retrieval.py\n"+__version__)
        print("-"*79)
        print("ARGS:")
        print("FlickrLogos root dir (--flickrlogos):")
        print("  > '"+options.flickrlogos+"'")
        print("Directory with result files (--indir):")
        print("  > '"+options.indir+"'")
        print("Output file ( --output):")
        print("  > '"+options.output+"'")
        print("-"*79)

    if options.flickrlogos is None or options.flickrlogos == "":
        print("Missing argument: --flickrlogos=<FlickrLogos-32 root directory>. Exit.")
        exit(1)

    if options.indir is None or options.indir == "":
        print("Missing argument: --indir=<directory with result files>. Exit.")
        exit(1)

    #==========================================================================
    # if output is a file and not "-" then all print() statements are redirected
    # to *both* stdout and a file.
    #==========================================================================
    if options.output is not None and options.output != "" and options.output != "-":
        if isdir(options.output):
            print("Invalid argument: Arg --output must denote a file. Exit.")
            exit(1)

        Tee(options.output, "w")

    #==========================================================================
    # compute scores
    #==========================================================================
    fl_eval_retrieval(options.indir, options.flickrlogos, options.verbose)

