import numpy as np
from PIL import Image
from keras.models import load_model
import warnings
import logging
import os
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)
warnings.filterwarnings("ignore")
model = load_model('model33.h5')
classes = {
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',
    5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)',
    7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',
    9:'No passing',
    10:'No passing veh over 3.5 tons',
    11:'Right-of-way at intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No vehicles',
    16:'Veh > 3.5 tons prohibited',
    17:'No entry',
    18:'General caution',
    19:'Dangerous curve left',
    20:'Dangerous curve right',
    21:'Double curve',
    22:'Bumpy road',
    23:'Slippery road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic signals',
    27:'Pedestrians',
    28:'Children crossing',
    29:'Bicycles crossing',
    30:'Beware of ice/snow',
    31:'Wild animals crossing',
    32:'End speed + passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep right',
    39:'Keep left',
    40:'Roundabout mandatory',
    41:'End of no passing',
    42:'End no passing veh > 3.5 tons'
          }

# The function of the image prediction
def classify(image_path):
    height = 32
    width = 32
    channels = 3
    image = Image.open(image_path)
    img = image.resize((height, width))
    img = np.array(img) / 255.
    img = img.reshape(1, height, width, channels)
    try:
        pred = model.predict_classes(img)[0]
        sign = classes[pred]
        answer = sign
    except():
        answer = 'Error loading image'
    return answer


# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

MARK, PHOTO, RESULT = range(3)


def start(update: Update, _: CallbackContext) -> int:
    reply_keyboard = [['Send photo']]

    update.message.reply_text(
        'Hi! My name is Road Sign Plot Bot. I can try to recognize the class of traffic sign in your photo.\n'
        'This is plot version and I can be wrong.'
        'Send /cancel to stop talking to me.\n\n'
        'Send /info for some information about bot \n'
        'Would you like to try?',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )

    return MARK


def info(update: Update, _: CallbackContext) -> int:
    user = update.message.from_user
    logger.info("User %s asked the info.", user.first_name)
    update.message.reply_text(
        'The bot can recognize 43 classes of the german road sign. Single-image, multi-class classification problem.\n'
        'The Kaggle competition: GTSRB - German Traffic Sign Recognition Benchmark\n' 
        'One photo should have one label of the sign'
    )

    return MARK


def mark(update: Update, _: CallbackContext) -> int:
    user = update.message.from_user
    update.message.reply_text(
        'Send a photo',
        reply_markup=ReplyKeyboardRemove(),
    )

    return PHOTO


def photo(update: Update, _: CallbackContext) -> int:
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    image_path = os.getcwd() + "/user_photo.jpg"
    reply = classify(image_path)
    logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
    update.message.reply_text(
        'I think the class of the photo is {}.'.format(reply)
    )

    return PHOTO


def skip_photo(update: Update, _: CallbackContext) -> int:
    user = update.message.from_user
    logger.info("User %s did not send a photo.", user.first_name)
    update.message.reply_text(
        'Maybe next time...'
    )

    return RESULT


def result(update: Update, _: CallbackContext) -> int:
    user = update.message.from_user
    logger.info("Bio of %s: %s", user.first_name, update.message.text)
    update.message.reply_text('Thank you! I hope we can talk again some day.')

    return ConversationHandler.END


def cancel(update: Update, _: CallbackContext) -> int:
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text(
        'Bye! I hope we can talk again some day.', reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def main() -> None:
    # Create the Updater and pass it your bot's token.
    updater = Updater("1757146019:AAE5eyK5QFgsKE6Rjh5DPzS2mW9zN6svE18")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MARK: [MessageHandler(Filters.regex('^(Send photo|)$'), mark),  CommandHandler('info', info)],
            PHOTO: [MessageHandler(Filters.photo, photo), CommandHandler('skip', skip_photo)],
            RESULT: [MessageHandler(Filters.text & ~Filters.command, result)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
