#!/usr/bin/env python
# Python function for text search and replace
from __future__ import print_function
import os, sys
usage = "usage: %s search_text replace_text [infile [outfile]]" % os.path.basename(sys.argv[0])

if len(sys.argv) < 3:
       print(usage)

else:

       # Get arguments
       stext = sys.argv[1]
       rtext = sys.argv[2]
       input = sys.stdin
       output = sys.stdout
       if 0:
              stext = '/usr/bin/env python' 
              rtext = '/usr/bin/env python\nfrom __future__ import print_function' 
       print("Converting string %s to %s..." % (stext,rtext))

       # Read input to memory as a string
       if len(sys.argv) > 3:
              input = open(sys.argv[3])
              print( "  input file %s" % sys.argv[3])
       s = input.read()
       if len(sys.argv) > 3:
              input.close()

       # Convert text and write output
       if len(sys.argv) > 4:
              output = open(sys.argv[4], 'w')
       output.write(s.replace(stext, rtext))
       if len(sys.argv) > 4:
              output.close()
