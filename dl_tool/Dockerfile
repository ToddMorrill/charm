FROM ruby:latest
ARG urlfile
ENV list=$urlfile
RUN apt-get update && apt-get install -y ffmpeg
RUN gem install open3
RUN apt-get install -y python3
RUN apt-get install -y pip
RUN python3 -m pip install -U yt-dlp
COPY process_urls.rb .
COPY $list .
COPY ldcc ldcc
ENTRYPOINT ["ruby", "process_urls.rb"]
CMD [$list]
