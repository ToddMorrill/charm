#!/usr/bin/env ruby

require_relative '../lib/ldcc'

USAGE = "Usage:  #{File.basename($0)} [-o out_dir] [field=value ...] [path/]data.file ...\n  default out_dir is '.' (CWD)"

opath = "."
if ARGV.size > 2 and ARGV[0] == '-o'
  abort "#{ARGV[1]} is not a directory\n#{USAGE}" unless File.directory?( ARGV[1] )
  ARGV.shift
  opath = ARGV.shift
end
hdr_fields = ARGV.select{ |str| str =~ /^\w+=\S.*/ }.map{ |str| str.split(/=/,2) }.to_h
files = ARGV.select{ |str| File.file?( str ) }

abort USAGE if files.size == 0 or files.size + hdr_fields.size != ARGV.size
abort "DO NOT USE THIS ON *.ldcc FILES" if files.select{|fn| fn =~ /\.ldcc\z/}.size > 0

files.each do |ifile|
  ofile = opath + "/" + File.basename( ifile ) + ".ldcc"
  ldcc = Ldcc.new( ifile )
  ldcc.put_into_hdr( hdr_fields ) if hdr_fields.size > 0
  ldcc.write( ofile )
end

=begin

= NAME

ldcc-create.rb

= SYNOPSIS

  ldcc-create.rb  [-o outdir]  [field=value ...]  [path/]data.file ... 

= DESCRIPTION

This command-line utility creates LDCC files from original data files.
It automatically sets values for the four mandatory header fields
(data_uid, data_type, data_bytes, data_md5), computes the byte count
of the header block (some multiple of 1024), formats the header
content as a YAML string (padded at the end with spaces as needed and
bounded by initial and final signature strings), and then creates a
new file by appending ".ldcc" to the input file name, writing the
header block, and then appending the data from the input file.

By default, the header contains only the four mandatory fields: two of
these (data_bytes, data_md5) are based on the input file's content,
and the other two (data_uid, data_type) are set to be the base name
and final extension of the input file name (e.g. for input file
"jpg/12345A.jpg", data_uid="12345A" and data_type="jpg").

Include one or more tokens of the form "key=value" as command-line
arguments to include additional fields with string or numeric values.

Multiple input file names can be supplied.  In this case, any/all
"key=value" settings supplied on the command line will be applied
identically to all output files.  (The command line can have input
names and "key=value" tokens interspersed in any order; the relative
ordering of "key=value" and file name args is ignored.)

By default, each new output ldcc file is created in the user's current
working directory.  Use the '-o out_dir' to create the files in some
other path.

=end
