# Creating a Test Environment
## Set up CCU Sandbox
```
# run the following command to obtain your IP address and add it to ./sandbox/ccu-config.yaml
python -c "import socket; print(f'IP is {socket.gethostbyname(socket.gethostname())}')"

# export the path to this config file
# or add it to your ~/.bashrc (or equivalent) profile
export CCU_SANDBOX=/home/iron-man/Documents/charm/integration/sandbox
```

## Create some mock data with Whisper
```

```

## Publishing mock data to the stream
Generate a jsonl message file in the expected format.
- Use Source2stream4ccu
Run python script.py --jsonl jsonlfile to inject messages in the CCU environment

TODO: figure out how to save jsonl output files
- Use Source2stream4ccu


## Subscribing to the mock data

## Testing
This gets into a long and complex discussion about how much testing is the right amount and in what situations. My
basic answer is this:
- All tests should be documented in your YAML file, no matter how many there are.
- All of your tests should run in 5-10 minutes or less.
- At least one test should run “real data” or as close to “real data” as possible.
- At least one test should run badly formed data, and therefore give an error message, not good results.
- If your container works with many data formats, test each one at least once.
- If it generates different types of results, make sure it generates them all sometime during testing.
- If your container works with either a CPU or a GPU, then test both.