#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import sys, os, re
from optparse import OptionParser

def compress_convert(f_in, f_out):

  # Read binary, header, then dimensions
  h_raw = f_in.read(5)
  binary = h_raw[0:2].decode()
  filetype = h_raw[2:5].decode()
  if not (binary == '\0B'):
    raise UnknownMatrixHeader("Non-binary kaldi filetype")
  globmin = None
  globrange = None
  if filetype == 'CM ':
    # Read global header of format struct'
    global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')])
    cm_info = f_in.read(16)
    globmin, globrange, rows, cols = np.frombuffer(cm_info, dtype=global_header, count=1)[0]
  else:
    raise UnknownMatrixHeader("The header contained '%s'" % header)

  # Write header
  if key != '' : f_out.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
  f_out.write('\0B'.encode()) # we write binary!
  f_out.write('CMT'.encode()) # compress transpose
  f_out.write(cm_info)

  # Read and write column headers
  col_headers = f_in.read(cols*8)
  f_out.write(col_headers)

  # Read, transpose, and write compressed data bytes
  data = np.reshape(np.frombuffer(f_in.read(cols*rows), dtype='uint8', count=cols*rows), newshape=(cols,rows)) # stored as col-major
  f_out.write((data.T).tobytes())

def read_key(fd):
  """ [key] = read_key(fd)
   Read the utterance-key from the opened ark/stream descriptor 'fd'.
  """
  key = ''
  while 1:
    char = fd.read(1).decode("latin1")
    if char == '' : break
    if char == ' ' : break
    key += char
  key = key.strip()
  if key == '': return None # end of file,
  assert(re.match('^\S+$',key) != None) # check format (no whitespace!)
  return key


# Main function: convert list and find new speech file paths
if __name__ == '__main__':
    
  # Parse input command line options
  parser = OptionParser()

  parser.add_option("-i", "--input", type="string", help="Input compressed archive", metavar="FILE")
  parser.add_option("-o", "--output", type="string", help="Output transpose compressed archive", metavar="FILE")
  (Options, args) = parser.parse_args()

  infile_name = Options.input
  outfile_name = Options.output
  f_in = open(infile_name,'rb')
  f_out = open(outfile_name,'wb')

  try:
    key = read_key(f_in)
    while key:
      print("Transposing compressed features %s" % key)
      compress_convert(f_in, f_out)
      key = read_key(f_in)
  finally:
    f_in.close()
    f_out.close()

