# there are issues in the create encodings, create_bulk_encodings functions (these are only used for creating encodings for new images.)
import streamlit as st
import face_recognition as fr
import os, cv2, ast
import numpy as np
import time

############################################ creating Backend

def get_time():
    current_time = time.localtime()
    year = current_time.tm_year
    month = current_time.tm_mon
    day = current_time.tm_mday
    hour = current_time.tm_hour
    minute = current_time.tm_min
    second = current_time.tm_sec
    return f'{year}-{month}-{day}-{hour}-{minute}-{second}\n\n'

def write_in_logs(text):
    with open('logs.txt', 'a') as lf: lf.write(text)
    return 0

def bgr_to_rgb(bgr_image):
    '''Takes a bgr image (cv2 works with) and returns RGB form of the image (face_recognition requires.) '''
    print('Inside bgr_to_rgb. Converting image to rgb format.')
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

def capture_img():
    print('Inside capture_img()')
    camera = cv2.VideoCapture(0)
    success, frame = camera.read()
    camera.release()
    print('-- Image captured successfully. Returning [success, frame]')
    return [success, frame]

def create_bulk_encodings():
    '''Needs to be called explicitly. Grabs all the images present inside the images folder, create encodings to all the images present inside and saves the encodings in separate files inside the encodings folder.'''
    print('Inside create_bulk_encodings')
    images_dir = 'images'

    # calling create encodings function for each image
    for image_file in sorted(os.listdir(images_dir)):
        # creating image path for passing
        image_path = os.path.join(images_dir, image_file)
        
        if os.path.isfile(image_path):
            # calling the function
            create_encodings(image_path = image_path, save = True)
            print(f'-- Encodings saved for {image_file}')
        else:
            print(f"{image_path} is not a file.")
            

# Creates provided image's 1st face encoding and saves or returs them. Used for creating encodings for train or test image.
def create_encodings(image_path = None, image = None, save: bool = False):
    '''Takes image and image_path, detects faces by loading the image_path if image is None else converts the image to rgb and then saves the 1st face encoding in encodings/image-name.txt and return 0, if {save} parameter is provided as True. else don't saves and return the 1st face encoding. Returns 0 for successful save and 1, if no face detected. Note: The image should contain only 1 face.'''
    print('Inside create_encodings')
    # defining useful variables
    print('-- Preparing Encodings..')
    encodings_folder = 'encodings'

    # loading image and detecting faces
    # if the image is provided then converting it to rgb.
    #------------ If the image is None, then the intention will be creating encodings and saving the file else just creating encodings and returning them. So in that case the image_path will also be none.
    if image is None and image_path is not None:
        print('-- Only image path is provided. Loading image and creating encoding_filename to use for saving the encoding.')
        image = fr.load_image_file(image_path)
        encoding_filepath = image_path.split('.')[0] + '-encoding.txt'
    else:
        print('-- Converting provided image to RGB..')
        image = bgr_to_rgb(image)

    # detecting face
    print('-- Image prepared successfully. Encoding it (detecting face.)')
    encoded_face = fr.face_encodings(image)
    # if we detected some faces, then saving encodings or returning them as np.array
    if len(encoded_face) != 0:
        print('-- Face detection successful. Converting face encodings to np.array.')
        # converting encoded_face to np.array (must)
        encoded_face = np.array(encoded_face[0], dtype = np.float64)
        # if we have to save the encodings
        if save == True:
            # going inside the encodings_folder
            print('-- Saving encodings')
            print(f'-- Going inside {encodings_folder} folder (creating it if not exists)')

            # if the encodings folder is not present then creating it.
            if not os.path.exists(encodings_folder):
                # os.chdir(encodings_folder)
                os.mkdir(encodings_folder)
            
            encoding_filepath = os.path.join(encodings_folder, encoding_filepath)
                
            print(f'-- Inside {encodings_folder} folder. Saving encoding')
            
            # Writting the encodings into the file.
            with open(encoding_filepath, 'w') as ef:
                print(f'-- Opened {encoding_filepath} for writting encodings.')
                # converting np array to list (reserving commas) and then str and writting into .txt file.
                ef.write(str(list(encoded_face)))     # picking the 1st encoded face
            print(f"-- Encoding written in {encoding_filepath}. Changing directory and returning 0")
            
            # os.chdir('..')

            return 0
        
        else:
            print('-- Returning encoded face.')
            return encoded_face
    
    # if no face is detected
    else:
        print('-- No face detected. Returning 1')
        return 1

# loads all the encoding files present inside the provided directory and returns a list of two dictionaries i.e. known_faces_encodings, known_faces_names.
def load_encodings(dir):
    '''loads all the encoding files present inside the provided directory and returns a list of two lists i.e. known_faces_encodings, known_faces_names.'''
    print('Inside load_encodings()')
    # used vars
    known_faces_encodings = []
    known_faces_names = []

    # grabbing all the files inside the directory
    print('-- Loading files..')
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        
        print(f'-- Reading {file}')
        with open(file_path, 'r') as f:
            
            # Extracting name of person
            name = ''
            # as the filenames are like shaukat-ali-khan-encodings.txt, so taking all the words except last one i.e. encoding
            filename = file.split('-')[:-1]
            # joining the words to beautify the name
            for file_name_part in filename:
                name += f'{file_name_part} '
            # capitilizing the name
            name = name.capitalize()
            print(f'-- Person name is: {name}')
            
            # reading file. FACED ISSUE IN THIS PART (THE DTYPE OF THE FILES CONTENT)
            encoding = f.read()
            # converting data to correct list
            encoding = ast.literal_eval(encoding)
            # converting encoding to numpy array with float64
            encoding = np.array(encoding, dtype = np.float64)
            print(f'-- Person encoding evaluated and converted to np.array with dtype float64. Dtype:', type(encoding))
            # appending the encoding
            known_faces_encodings.append(encoding)
            known_faces_names.append(name)
            print(f'-- Appended {name} encoding and its name to the relevant lists.')

        print(f'-- {file} processed.')

    print('-- Returning list: [known_faces_encodings, known_faces_names]')
    print(f'-- Known faces: {known_faces_names}')
    write_in_logs(f'Data loaded at {get_time()}')
    return [known_faces_encodings, known_faces_names]

# compares test and known encodings and  return name of known person (if the test person is known) else return unknown
def compare_encodings(known_encodings, known_faces, test_encoding):
    '''Compares test and known encodings and  return name of known person (if the test person is known) else return unknown'''
    print('\nInside compare_encodings()')
    print('-- Type input encoded face..', type(test_encoding))
    print('-- Type of a face from known encoded faces..', type(known_encodings[0]))
    print('-- Comparing faces')
    comparison_results = fr.compare_faces(known_encodings, test_encoding, tolerance = 0.5)
    
    # if the face is known, grabbing 1st known face name
    print('-- Looking for match')
    if True in comparison_results:
        print('-- Match found. Extracting person name.')
        match_index = comparison_results.index(True)
        name = known_faces[match_index]
        print(f'Matched person name is: {name}. Returning it.')
        return name
    
    else:
        print('-- No match found. Returning {unknown}')
        return 'unknown'


########################################### Creating front-end

def main():
    ############################################ configuration
    print('Setting page configurations i.e. title, session_state var')
    st.set_page_config(page_title = 'Verify User', page_icon = 'camera', layout = 'centered')
    session = st.session_state
    # time = time()

    # setting first_load var in order to not load the encodings again and again (with each click)
    print('Defining first_load, known_faces_encodings, known_faces_names variables in session state.')
    if 'first_load' not in session:
        session.first_load = True
    else:
        session.first_load = False

    # vars to store the encodings
    if 'known_faces_encodings' not in session:
        session.known_faces_encodings = None
    if 'known_faces_names' not in session:
        session.known_faces_names = None

    ############################################ creating UI
    st.title('Verify User :camera:')
    st.button('Capture And Verify', key = 'verify_btn', type = 'primary')
    st.divider()

    ########################################### Processing
    # loading known data
    if session.first_load:
        print('\n--------------------------------------------------------------------------------- \
              \nsession.first_load indicated page is loading for the first time. Loading known encodings.')
        session.known_faces_encodings, session.known_faces_names = load_encodings('encodings')
        print('-- Success')

    # setting variables
    captured_img_name = 'captured-image.jpg'

    # if verify btn is clicked, then capturing image and verifying it.
    if session.verify_btn:
        print('\n--------------------------------------------------------------------------------- \
              \nVerify btn clicked. Processing..')
        with st.spinner('Processing..'):
            success, frame = capture_img()

            # if we got a frame
            if success:
                print('\n--------------------------------------------------------------------------------- \
              \nImage captured successfully. Creating encoding..')

                # creating encoding
                captured_img_encoding = create_encodings(image = frame)

                # if a face is detected, then comparing it. Using int instance to verify that the encoding was successfully created because the function return 0 it successfully saves the encodings and 1 if no face detected and the encodings if the encoding was successfully returned.
                if not isinstance(captured_img_encoding, int):
                    print('\n------------------------------------------------------------------------------ \
                    \nSuccess. Comparing for match..')
                    result = compare_encodings(session.known_faces_encodings, session.known_faces_names, captured_img_encoding)
                    print('\n------------------------------------------------------------------------------ \
                    \nComparison completed.')
                    if result != 'unknown':
                        st.subheader(f":man-raising-hand: Hi :blue[{result}].")
                        st.caption('Welcome to your LandLedger.')
                        # saving in logs
                        print('Recording in logs..')
                        write_in_logs(f'{result} logged in at {get_time()}')
                    else:
                        st.caption(":lock: Sorry, You don't have access! Please contact the owner at owner@example.com.")
                        # saving in logs
                        print('Recording in logs..')
                        write_in_logs(f'{result} tried to log in at {get_time()}')

                elif captured_img_encoding == 1:
                    st.caption(":no_entry_sign: Sorry, No face detected!")
                    # saving in logs
                    print('Recording in logs..')
                    write_in_logs(f'No face detected for try made at: {get_time()}')

                else:
                    print("\n------------------------------------------------------------------------------ \
                    \nCreate encodings function returned something other that encodings or 1.")
                    write_in_logs(f'Create encodings returned 0 for try made at: {get_time()}')

# calling main function
if __name__ == '__main__':
    main()

# create_bulk_encodings()