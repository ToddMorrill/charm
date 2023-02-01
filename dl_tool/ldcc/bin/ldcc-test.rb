#!/usr/bin/env ruby

require_relative '../lib/ldcc'

USAGE = "#{File.basename($0)} [path/]file.ldcc ..."

files = ARGV.select{ |str| str =~ /\.ldcc\z/ and File.file?( str ) }

abort USAGE if files.size == 0 or files.size != ARGV.size

files.each do |ifile|
  begin
    problems = Ldcc.hdr_check( ifile )
    report = ( problems.size == 0 ) ? "ok" : problems.join( " " )
  rescue => e
    report = e.message
  end
  puts [ ifile, report ].join("\t: ")
end

=begin

= NAME

ldcc-test.rb

= SYNOPSIS

  ldcc-test.rb  [path/]file.ldcc ...

= DESCRIPTION

This command-line utility checks for problems in the header block of
one or more LDCC files provided as command-line arguments.

For each file being tested, a single line of output is printed to
stdout to report the test results.  Each line begins with the file
name (including directory path, if provided), followed by a tab, a
colon, and a space.

If no problems are detected, the output line ends with "ok";
otherwise, the line includes one or more of the following terms -- if
more than one, these are space-separated:

  - data_md5:mismatch -- value differs from actual md5 of data
  - data_md5:missing  -- header lacks a required field...
  - data_uid:missing
  - data_type:missing
  - data_bytes:missing
  - data_bytes:bad_value -- value isn't numeric, or != actual size of data

=end
