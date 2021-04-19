#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == '__main__':
    usage = '[options] output_model input_model1 input_model2 ... input_modelN'
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("--modulus", dest="modulus", default=1, type=int)
    parser.add_argument("--num-gpus", dest="num_gpus", default=1, type=int)
    args = parser.parse_args()

    eprint("modulus=%d num_gpus=%d" % (args.modulus, args.num_gpus))

    if "SGE_HGR_gpu" in os.environ or "SGE_HGR_gpu_card" in os.environ:
        if "SGE_HGR_gpu" in os.environ:
            gpus = os.environ["SGE_HGR_gpu"].split(" ")
            eprint("Found SGE_HGR_gpu : %s" % os.environ["SGE_HGR_gpu"])
        if "SGE_HGR_gpu_card" in os.environ:
            gpus = os.environ["SGE_HGR_gpu_card"].split(" ")
            eprint("Found SGE_HGR_gpu_card : %s" % os.environ["SGE_HGR_gpu_card"])

        if args.modulus == 1:
            eprint("Setting GPUs to %s" % ",".join(gpus))
            print("%s" % ",".join(gpus))
        else:
            if len(gpus) == 1:
                eprint("Setting GPUs to %s" % ",".join(gpus))
                print("%s" % ",".join(gpus))
            else:
                if args.num_gpus > 1:
                    if args.num_gpus > len(gpus):
                        eprint("Error: num_gpus is greater than gpus supplied in env variable (%d > %d)" % (args.num_gpus, len(gpus)))
                        exit(1)

                    # split GPUs into num_gpu groups
                    gpu_groups = [gpus[i:i + args.num_gpus] for i in range(0, len(gpus), args.num_gpus)]
                    index = args.modulus % len(gpu_groups)
                    eprint("Setting GPUs to %s" % ",".join(gpu_groups[index]))
                    print("%s" % ",".join(gpu_groups[index]))
                else:
                    index = args.modulus % len(gpus)
                    eprint("Setting GPU to %s" % gpus[index])
                    print("%s" % gpus[index])
    else:
        print("-1")

    exit(0)
