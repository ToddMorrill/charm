# ldcc
Tools to create, and validate LDCC headers for data files

An LDCC header is a block of text that is prepended to any given data
file, and provides "metadata" about the file.  The basic, essential
properties of the header are:

 - It contains only UTF8-encoded text data.
 - It's size is always a multiple of 1024 bytes.
 - It always begins with a 16-byte sequence:
   -- "LDCc   \n1024   \n" (NB: 3 spaces on each of two lines), or
   -- "LDCc   \n2048   \n" (if metadata content >1K and <2K bytes) *
 - It always ends with the 8-byte sequence: "endLDCc\n"
 - Between those initial and final strings is a YAML-encoded hash **

Notes:

 - (*) For metadata content > (1K * n) bytes and < (1K * (n+1)), the
   number in the second line (and the byte count of the header block
   as a whole) must be (1024 * (n+1)).

 - (**) The YAML-encoded hash is a simple set of "key: value" tuples;
   after the YAML string has been built from the hash, it is padded
   with spaces so that, when combined with the initial (16-byte) and
   final (8-byte) strings, the byte length of the full string is a
   multiple of 1024.

The software in this package is organized into a single library
module (lib/ldcc.rb), and executable scripts in the bin/
directory for creating and validating LDCC headers.
