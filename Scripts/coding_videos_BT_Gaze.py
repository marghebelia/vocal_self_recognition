saving_dir = '/Users/fannie/Documents/SOABB_ETUDE1/codage_video/Babbling_task/coded_BT_gaze/'  # adresse où vous voulez sauver vos données
video_dir = '/Users/fannie/Documents/SOABB_ETUDE1/codage_video/Babbling_task/'  # adresse où sont vos vidéos

import numpy as np
import pandas
import cv2
import os, glob
import time
import sys
from psychopy import visual, core, event


pandas.options.display.max_columns = 500
pandas.options.display.max_rows = 500

# Get list of all video files in the directory
video_files = glob.glob(os.path.join(video_dir, "*.mp4"))  # Adjust the file extension as needed

# Créer une fenêtre PsychoPy
w = visual.Window(size=(400, 700), pos=(880, 0), fullscr=False, allowGUI=False, monitor='testMonitor', units='pix', color=[121, 121, 121], colorSpace='rgb255')
mouse = event.Mouse(visible=True, win=w)
mouse_pos = mouse.getPos()

# Définir le texte et sa position
text_coding_lines = [
    "What is doing the BB ?",
    "",
    "Away (A)",
    "Mum (Z)",
    "Object (E)",
    "Unsure (R)",
#    "Look + touch/reach mirror (O)", 
#    "Look away + touch/reach mirror (P)",
    "Blink (B)",
    "Break (N)",
    "", 
    "Other keys:",
    "",
    #"play_key (space)",
    "Rewind (I)",
    "Faster (P)",  
    "Uncodable (enter)"
]

# Créer les objets TextStim pour chaque ligne de texte
text_stimuli = []
y_pos = 285
for line in text_coding_lines:
    if line == "What is doing the BB ?" or line == "Other keys:":
        color = 'white'
        bold = True
    else:
        color = 'BlanchedAlmond'
        bold = False
    
    # Créer un objet TextStim pour chaque ligne de texte
    text_stimuli.append(visual.TextStim(w, text=line, pos=(0, y_pos), wrapWidth=350, height=26, color=color, bold=bold))
    y_pos -= 40  # Espacement entre les lignes

# Dessiner les textes et afficher la fenêtre
for stim in text_stimuli:
    stim.draw()
w.flip()


# Function to rewind the video
def rewind_video():
    global where_FR, df, tri, ti #, phase
    where_FR = max(0, where_FR - rewind)  # Ensure frame number doesn't go negative
    ti = where_FR  # Update timing based on new frame position
    df = df[df["timing"] < ti]  # Erase previous annotations after this point
    if not df.empty:
        # phase = df.loc[df["where_FR"] == df["where_FR"].max(), "phase"].values[0]
        tri = df.loc[df["where_FR"] == df["where_FR"].max(), "trial"].values[0]
    else:
        # phase = None
        tri = None
    print(f"Rewinding to frame: {where_FR}, trial: {tri}") #, phase: {phase}
    cap.set(cv2.CAP_PROP_POS_FRAMES, where_FR)  # Set the video position


# opens video
for video_path in video_files:
    video_ID = os.path.split(video_path)[1]
    print(f"Processing video: {video_ID}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        continue
    if not video_files:
        print("Aucune vidéo trouvée dans", video_dir)
        sys.exit()
  
    # info about video
    frame_rate = cap.get(cv2.CAP_PROP_FPS) # nb of frames per sec > should be 50 fps ; I CHANGED IT FROM INT TO REAL VALUE
    len_frame = 1/frame_rate # length of a frame in seconds > should be 20ms sample size
    nb_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Number of frames in the video file.
    length_sec = nb_frames/frame_rate
    rewind = 5 # to rewind X frames
    rewind_sec = rewind/frame_rate
    where = cap.get(cv2.CAP_PROP_POS_MSEC) # Current position of the video file in milliseconds or video capture timestamp
    where_FR = cap.get(cv2.CAP_PROP_POS_FRAMES) # Current position of the video file in frames
    height = cap.get(4)
    width = cap.get(3)
    print("frame_rate:", frame_rate, " - total_nb_frames:", nb_frames, " - length_sec:", length_sec,
      " - where we are now:", where, where_FR, " - height:", height, " - width:", width)

    if height < 500: # TO RESIZE IF TOO SMALL
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, (height*2))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, (width*2))
        height = cap.get(4)
        width = cap.get(3)
    print("new heigh/width: ", height, width)

    # parameters
    max_write = int(frame_rate*5) # to write every X frames or X seconds to be faster
    print("writes every", max_write, "frames")
    #phase = 'codable' 
    # Initializes values for coding the video / iterating
    tri = 0 # trial nb
    ti = 0 # time frame > starts with zero > first frame is the onset/beginning
    aga = 0 # to not write .csv every time but only every max_write seconds rather than at each time frame
    ## Creates keylist: mettre dans cette liste les numéros de clefs à utiliser pour votre codage; 
    #### exemple: pour la barre espace sur un clavier azerty c'est: 32
    used_keys = np.array([97, 122, 101, 114, 98, 110, 2, 3, 13, 27])
    key_names = pandas.DataFrame({"key_nb": used_keys, "key_name": np.repeat("nan", len(used_keys))})

    #### DANS CETTE PARTIE, vous pouvez définir des "key_name" à sauver quand vous appuyez sur des clefs particulières
    ## Qu'est-ce que BB regarde ?
    key_names.loc[key_names["key_nb"] == 97, "key_name"] = "A" # A for Away
    key_names.loc[key_names["key_nb"] == 122, "key_name"] = "M" # Z for Mum
    key_names.loc[key_names["key_nb"] == 101, "key_name"] = "O" # E for Object
    key_names.loc[key_names["key_nb"] == 114, "key_name"] = "U" # R for Unsure
    #key_names.loc[key_names["key_nb"] == 111, "key_name"] = "Tm" # O for look and touch or reach the mirror
    #key_names.loc[key_names["key_nb"] == 112, "key_name"] = "Ta" # P for look away and touch or reach own face
    key_names.loc[key_names["key_nb"] == 98, "key_name"] = "Blink" # B - when bb blinks
    key_names.loc[key_names["key_nb"] == 110, "key_name"] = "Break" # N - when the baby needs a beak in the experiment (without headphones)

    ## UTILITY
    #play_key = 32 # space
    rewind_key = 2 # I (for windows - (I) = 105)
    fast_key = 3 #  (for windows - (P) = 112)
    uncodable = 13 # Enter
    #key_names.loc[key_names["key_nb"] == play_key, "key_name"] = "PLAY" 
    key_names.loc[key_names["key_nb"] == rewind_key, "key_name"] = "CANCEL" # return 5 frames before when there was an error
    key_names.loc[key_names["key_nb"] == fast_key, "key_name"] = "uncodable" # goes 1 sec by 1 sec > to use when there is nothing happening
    key_names.loc[key_names["key_nb"] == uncodable, "key_name"] = "uncodable" # uncodable = when it's impossible to determine where the BB/mum looks or points or reaches etc... (e.g., something blocks the view or the direction of the gaze is ambiguous)

    ### CHECK IF FILE ALREADY EXISTS AND if so STARTS AGAIN FROM THE NEXT FRAME
    past_files = np.array(glob.glob(saving_dir + video_ID + "*"))
    print("past_files", past_files)
    if past_files.size > 0:
        df = pandas.read_csv(past_files[0], sep = ";")
        ti = np.nanmax(df["timing"]) + 1 # we start at the next frame
        tri = np.nanmax(df["trial"]) # we keep the trial number
        #phase = df.loc[len(df)-1, "phase"] # we take the last value of phase
        cap.set(cv2.CAP_PROP_POS_FRAMES, ti)
        print("file already exists, we start again at frame n°", ti, ", trial n°", tri)
    else:   # Creates a new pandas df 
        df = pandas.DataFrame({"video": np.repeat(video_ID,1), "timing": np.repeat(np.nan,1), "timing_sec": np.repeat(np.nan,1), 
                       "trial": np.repeat(0,1), # "phase": np.repeat(phase,1), 
                       "code": np.repeat("starting",1), "key_name" : np.repeat("starting",1), 
                       "FPS": np.repeat(frame_rate,1), "total_nb_frames": np.repeat(nb_frames,1),
                        "where_msec": np.repeat(where,1), "where_FR": np.repeat(where_FR,1)})

    df.head(2)


    ## LOOPING TO CODE THE VIDEO
    coding_starts = time.time()
    filename = video_ID + "_coded_" + str(int(coding_starts)) + ".csv"
    filename = os.path.join(saving_dir, filename)

    while ti < nb_frames: # tant qu'on est pas à la fin de la vidéo
        start = time.time()
        where = cap.get(cv2.CAP_PROP_POS_MSEC)
        where_FR = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        ti_sec = np.round(len_frame*ti, 4)
        print("frame n°:", ti, " = ", ti_sec, " secs. - trial n°:", tri, " - where in video? in ms:", where, " - in FR:", where_FR)
        if ret:
            cv2.startWindowThread()
            cv2.imshow('frame',frame)
            u = cv2.waitKey(0) ## WAITS FOR A KEYPRESS

            if np.isin(u, used_keys): 
                code_name = np.array(key_names.loc[key_names["key_nb"] == u, "key_name"])[0]
            else:
                code_name = "nan"

            #if u == play_key:  # P: for each play iterate trial number & update phase = uncodable
            #    tri = tri + 1
                #phase = 'uncodable'
                #print("pressed: ", u, code_name, " - phase?", 'uncodable')


            if u == rewind_key:  # left arrow key for rewind
                rewind_video()

            elif u == fast_key: # FAST FORWARD = reads 1 second per 1 second
                new = pandas.DataFrame({"video": video_ID, "timing": ti, "timing_sec": ti_sec, "trial": tri, #"phase": phase,
                                                'code': np.repeat(u,1), 'key_name': code_name, 
                                                "FPS": frame_rate, "total_nb_frames": nb_frames,
                                                "where_msec": where, "where_FR": where_FR})
                df = pandas.concat([df, new], ignore_index=True)
                where_FR += frame_rate # set where we want to go: where + nb of frames > 1 second
                ti += frame_rate # same for ti for the record
                #phase = 'uncodable'
                print("fast forwards to: ", where_FR, ti, "pressed: ", u, code_name) #, " - phase?", 'uncodable'
                cap.set(cv2.CAP_PROP_POS_FRAMES, where_FR) # actually jump there        else: # if we do not want to rewind/cancel > writes

            else: # if we do not want to rewind/cancel > writes
                print("pressed: ", u, code_name) # , " - phase?", phase
                #phase = 'codable'
                new = pandas.DataFrame({"video": video_ID, "timing": ti, "timing_sec": ti_sec, "trial": tri, #"phase": phase,
                                                'code': np.repeat(u,1), 'key_name': code_name,
                                                "FPS": frame_rate, "total_nb_frames": nb_frames,
                                                "where_msec": where, "where_FR": where_FR})
                df = pandas.concat([df, new], ignore_index=True)
                ti = ti + 1 # iterates frame number
                aga = aga + 1 # iterates nb of saving number


            if aga == max_write:
                print("writing to csv")
                df.to_csv(filename, sep=";", index = False)
                aga = 0


        if ((ti >= nb_frames) | (u == 27)): # nb_frames but now less to test / OR ESCAPE KEY PRESSED
            print("end of video or escaping - writing to csv")
            df.to_csv(filename, sep=";", index = False)

            # Close the window and clean up
            try:
                w.close()
                core.quit()
            except Exception as e:
                print("Error closing PsychoPy:", e)
            break
    cap.release()

    print("out of loop - saving")
    df.to_csv(filename, sep=";", index = False)

## THIS BUGS ON MAC FOR SOME REASON SO DON'T WORRY ABOUT CLOSING THE WINDOW...
cv2.waitKey(5)
cv2.destroyAllWindows()
print("tried to kill...")
cv2.destroyAllWindows()
cv2.waitKey(5)