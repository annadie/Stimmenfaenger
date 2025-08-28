if __name__ == '__main__':

    import os
    import sys
    from keybert import KeyBERT
    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()

    from RealtimeSTT import AudioToTextRecorder

    # Initialize the KeyBERT model
    model = KeyBERT('sentence-transformers/LaBSE')


    recorder = AudioToTextRecorder(
            spinner=False,
            silero_sensitivity=0.01,
                model="tiny",
            language="de",
            )

    print("Say something...")
    
    try:
        while (True):
            text=recorder.text()
            # Extract keywords
            keywords = model.extract_keywords(text)

            # Print the keywords
            print(text)
            print("Keywords:")
            for keyword in keywords:
                print(keyword)
            print("----------------------------")

    except KeyboardInterrupt:
        print("Exiting application due to keyboard interrupt")
