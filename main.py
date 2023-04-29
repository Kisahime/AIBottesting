import json
import requests
from pydub import AudioSegment
from pydub.playback import play
import io
from twitchio.ext import commands
import creds
from transformers import BloomTokenizerFast 
from petals import DistributedBloomForCausalLM
import torch

MODEL_NAME = "bigscience/bloom-petals"
tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME)
model = model.cuda()

def initVar():
    global EL_key
    global EL_voice
    global EL

    try:
        with open("config.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("Unable to open JSON file.")
        exit()

    class EL:
        key = data["keys"][0]["EL_key"]
        voice = data["EL_data"][0]["voice"]

async def call_api(message):
    with model.inference_session(max_length=512) as sess:
        while True:
            prompt = message.content
            if prompt == "":
                break
            prefix = f"Human: {prompt}\nArisa:"
            prefix = tokenizer(prefix, return_tensors="pt")["input_ids"].cuda()
            
            while True:
                outputs = model.generate(
                    prefix, max_new_tokens=1, do_sample=True, top_p=0.9, temperature=0.75, session=sess
                )
                outputs = tokenizer.decode(outputs[0, -1:])
                return outputs
                
                if "\n" in outputs:
                    break
                prefix = None  # Prefix is passed only for the 1st token of the bot's response                
        

async def TTS(outputs):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{EL.voice}"
    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": EL.key,
        "Content-Type": "application/json",
    }
    data = {
        "text": outputs,
        "voice_settings": {"stability": 0.75, "similarity_boost": 0.75},
    }

    response = requests.post(url, headers=headers, json=data)
    audio_content = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
    play(audio_content)

class Bot(commands.Bot):
    def __init__(self):

        super().__init__(
            token=creds.TWITCH_TOKEN, prefix="!", initial_channels=[creds.TWITCH_CHANNEL]
        )

    async def event_ready(self):
        print(f"Logged in as | {self.nick}")

    async def event_message(self, message):

        if message.echo:
            return
            
        response = await call_api(message)
        await TTS(response)

if __name__ == "__main__":
    initVar()
    print("\n\Running!\n\n")
    Bot().run()
