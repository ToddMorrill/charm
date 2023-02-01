#!/usr/bin/env ruby

require 'open3'

USAGE = "\nUsage: #{File.basename($0)} <input_list_file>\n"\
        "Input list file must be a tab-delimited list with the following columns, in order:\n"\
        "source_uid    file_uid    URL\n"\
        "Video will be downloaded as MP4, using the file_uid as name, and the original unwrapped version will be removed."

abort "#{USAGE}" if ARGV.size != 1 or !File.file?(ARGV[0])
listfile = ARGV[0]

def process_url(url, file_uid, src_uid, ofil, cmd)

  puts "Attempting to download #{url}..."
  stdout, stderr, status = Open3.capture3(cmd)
  if status == 0
    # if ofil exists, wrap it
    stdout, stderr, status = Open3.capture3("./ldcc/bin/ldcc-create.rb -o /tmp source_uid=#{src_uid} parent_uid=na has_siblings='false' root_uid=na data_url=#{url} data_uid=#{file_uid} /tmp/#{ofil}")
    if status == 0
      puts "Successfully downloaded #{url} and added wrapper. Output file is #{ofil}.ldcc."
      File.delete("/tmp/#{ofil}")
    else
      puts "Received error while wrapping #{ofil}: \n #{stderr}"
      File.delete("/tmp/#{ofil}")
    end
  else
    puts "Received error while processing #{ofil}: \n #{stderr}"
  end

end

File.readlines(listfile).map(&:chomp).each do |l|
  src_uid, file_uid, url = l.split("\t")
  next if l =~ /^source_uid/  #skip header line
  ofil = file_uid + '.mp4'
  if src_uid  == 'S0U' #youtube defaults to f22 mp4 first
    cmd = "yt-dlp --prefer-ffmpeg #{url} -f 22/mp4 -o /tmp/#{ofil}"
  else 
    # earliest h264 videos collected with default settings
    if file_uid < 'M01004B25'
      cmd = "yt-dlp --prefer-ffmpeg #{url} --merge-output-format mp4 -o /tmp/#{ofil}"
    else
      #specify lower quality to ensure we get h264 instead of the less compatible hevc, lower than max res for space
      cmd = "yt-dlp --prefer-ffmpeg #{url} -f 'worstvideo[width<=800]+bestaudio/worstvideo[width<=1200]+bestaudio/worstvideo[width<=1900]+bestaudio/worstvideo+bestaudio/mp4' -o /tmp/#{ofil}"
    end
  end
  process_url(url, file_uid, src_uid, ofil, cmd)
end
