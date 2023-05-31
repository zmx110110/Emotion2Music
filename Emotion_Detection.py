import argparse
import io
import cgitb
cgitb.enable(display=0, logdir="/path/to/logdir")
from google.cloud import vision
from google.cloud.vision import types


def main(image_file):
    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.face_detection(image=image)
    labels = response.face_annotations
    for label in labels:
        print('Joy Likelihood: {}'.format(label.joy_likelihood))
        print('Anger Likelihood: {}'.format(label.anger_likelihood))
        print('Sorrow Likelihood: {}'.format(label.sorrow_likelihood))
        print('Surprise Likelihood: {}'.format(label.surprise_likelihood))
        
    maximum = max(label.joy_likelihood,label.anger_likelihood,label.sorrow_likelihood,label.surprise_likelihood)

    if maximum == label.sorrow_likelihood:
        print('sad')
    elif maximum == label.joy_likelihood:
        print('happy')
    elif maximum == label.anger_likelihood:
        print('angery')
    else:
        print('surprise')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file', help='The image you\'d like to label.')
    args = parser.parse_args()
    main(args.image_file)
