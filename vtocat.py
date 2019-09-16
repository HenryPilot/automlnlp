#for playbackvoice()
import sounddevice as sd
from scipy.io.wavfile import write
from playsound import playsound

#for plotvoice() 
import matplotlib.pyplot as plot
from scipy.io import wavfile

#for ratevoice()
from scipy.io import wavfile
from pypesq import pypesq

#for reducenoise()
from logmmse import logmmse_from_file

#for s2t() #speech to text 
import io
import os
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

#for 
import sys
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

#================
#Global variables
#================

#the filename of the original voice recording 
voicewave='input1.wav'

#the filename of the voice recording after noise reduction processing to original voice recording 
voicerd='input1-rd.wav'

#the filename of the converted text of the voice recording
voicetext="s2text.txt"

########################
#record the sound
########################
def recordvoice():
    fs = 16000 # Sample rate
    seconds = 5  # Duration of recording

    startprompt=" "
    startprompt=str(input("ready to record?"))
    
    if startprompt !='y':
        print('Please press y and then enter key to proceed. Exiting...')
        exit(0)
    else:
        print("recording... please speak to the microphone, you have %d seconds recording time" % seconds)
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        write(voicewave, fs, myrecording)  # Save as WAV file

 
########################
# Plot the sound
########################
def plotvoice():
    samplingFrequency, signalData = wavfile.read(voicewave)
    plot.subplot(211)
    plot.title('Spectrogram')
    plot.plot(signalData)

    plot.xlabel('Sample')
    plot.ylabel('Amplitude')

    plot.subplot(212)
    plot.specgram(signalData,Fs=samplingFrequency)

    plot.xlabel('Time')
    plot.ylabel('Frequency')

    plot.show(block=True)


########################
#play back the sound 
########################
def playbackvoice():
    print("voice recorded and it's playing back now")
    playsound(voicewave)


########################
#rate the sound quality
########################
def reducenoise():
    logmmse_from_file(voicewave, output_file=voicerd, initial_noise=60, window_size=0, noise_threshold=0.15)



########################
#rate the sound quality
########################
def ratevoice():
    rate, ref = wavfile.read(voicewave)
    rate, deg = wavfile.read(voicerd)
    pesqvalnb=pypesq(rate, ref, deg, 'nb')
    print("The voice quality is %.2f (0=worst, 4.5=best)" % pesqvalnb)


########################
#speech to text
########################
def s2t(voicefile):
    # Instantiates a client
    client = speech.SpeechClient()

    print("Converting voice in file \"%s\" to text..." % voicefile)
    print("......")
    # Loads the audio into memory
    with io.open(voicefile, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='yue-Hant-HK')

    # Detects speech in the audio file
    response = client.recognize(config, audio)
    
    fout = open(voicetext, 'w', encoding="utf-16")

    for result in response.results:
        print('You said: {}'.format(result.alternatives[0].transcript))
        fout.write(' {}'.format(result.alternatives[0].transcript))
        fout.write('\n')
    
    fout.close()



########################
#Automl analyze text
########################
def automlcat():
    def get_prediction(content, project_id, model_id):
        prediction_client = automl_v1beta1.PredictionServiceClient()
        name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
        payload = {'text_snippet': {'content': content, 'mime_type': 'text/plain' }}
        params = {}
        request = prediction_client.predict(name, payload, params)
        return request  # waits till request is returned

    fin=open(voicetext,"r", encoding='utf-16-le')

    if fin.mode == 'r':
        content = fin.read()

    print("\nAnalyzing...\n" + content + "\n")

    project_id="nlapi-test-1567058483241"

    model_id="TCN581844099215013286" 

    print("The system guess the requested banking service is: \n")

    print(get_prediction(content, project_id,  model_id))


########################=================================================
#start the action main()
########################=================================================

recordvoice()
playbackvoice()
plotvoice()

rdprompt=" "
print("\n")
print("To trigger the background noice reduction process")
print("Please press y and then enter key ")
rdprompt=str(input("Or any other key to use original voice recording:"))
    
if rdprompt =='y':
    print("Background noise reduction is under processing...")
    reducenoise()
    s2t(voicerd)
else:
    print("Keep going without noise reduction processing...")
    s2t(voicewave)

automlcat()


