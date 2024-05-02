import melee
import pandas as pd
import os
import time
import re
import numpy as np
import tensorflow as tf
import pickle
import shutil

from tensorflow.keras.callbacks import EarlyStopping

        

def contains_fox_in_file(filename):
    return ('Fox' in filename and 'FD' in filename) or ('Fox' in filename and 'Final Destination' in filename)

def check_game_for_fox(filename):
    console = melee.Console(path=filename, system="file", allow_old_version=True)
    console.connect()
    gamestate=console.step()
    if len(gamestate.players.keys()) != 2:
        return False
    p1, p2 = gamestate.players.keys()

    return (gamestate.players[p1].character == melee.Character.FOX or gamestate.players[p2].character == melee.Character.FOX) and (gamestate.stage == melee.Stage.FINAL_DESTINATION) and (gamestate.players[p1].character == melee.Character.FALCO or gamestate.players[p2].character == melee.Character.FALCO)

def calculate_closest_m_direction(x,y):
    m_directions = {'north':[0.5,1],'north_east':[0.75,0.75], 'east':[1.0, 0.5],
                  'south_east':[0.75, 0.35], 'south':[0.5, 0], 'south_west':[0.35, 0.35],
                  'west':[0, 0.5], 'north_west': [0.35, 0.75]}
    
    closest_m_direction = ""
    min_distance = np.inf
    for m_direction, coordinate in m_directions.items():
        current_distance = np.sqrt((x-coordinate[0])**2 + (y-coordinate[1])**2)
        if current_distance < min_distance:
            min_distance = current_distance
            closest_m_direction = m_direction
    return closest_m_direction

def slippi_to_df(original_df, filename):
    console = melee.Console(path=filename, system="file", allow_old_version=True)
    current_df = pd.DataFrame()
    console.connect()
    fr = []


    #Player 1 data
    c_stick_1_X = []
    c_stick_1_Y = []
    c_north_1 = []
    c_north_east_1 = []
    c_east_1 = []
    c_south_east_1 = []
    c_south_1 = []
    c_south_west_1 = []
    c_west_1  = []
    c_north_west_1 = []
    A_1 = []
    B_1 = []
    X_1 = []
    Y_1 = []
    Z_1 = []
    main_stick_1_X = []
    main_stick_1_Y = []
    m_north_1 = []
    m_north_east_1 = []
    m_east_1 = []
    m_south_east_1 = []
    m_south_1 = []
    m_south_west_1 = []
    m_west_1  = []
    m_north_west_1= []
    l_shoulder_1 = []
    r_shoulder_1 = []

    #Player 2 data

    c_stick_2_X = []
    c_stick_2_Y = []
    c_north_2 = []
    c_north_east_2 = []
    c_east_2 = []
    c_south_east_2 = []
    c_south_2 = []
    c_south_west_2 = []
    c_west_2  = []
    c_north_west_2 = []
    A_2 = []
    B_2 = []
    X_2 = []
    Y_2 = []
    Z_2 = []
    main_stick_2_X = []
    main_stick_2_Y = []
    m_north_2 = []
    m_north_east_2 = []
    m_east_2 = []
    m_south_east_2 = []
    m_south_2 = []
    m_south_west_2 = []
    m_west_2  = []
    m_north_west_2 = []
    l_shoulder_2 = []
    r_shoulder_2 = []
    gamestate = console.step()
    if len(gamestate.players.keys())>2:
        return original_df

    p1, p2 = gamestate.players.keys()
    while True:
        gamestate = console.step()
        # step() returns None when the file ends
        
        
        if gamestate is None:
            break
        # Player 1 data
        
        
        A_1.append(int(gamestate.players[p1].controller_state.button[melee.Button.BUTTON_A]))
        B_1.append(int(gamestate.players[p1].controller_state.button[melee.Button.BUTTON_B]))
        X_1.append(int(gamestate.players[p1].controller_state.button[melee.Button.BUTTON_X]))
        Y_1.append(int(gamestate.players[p1].controller_state.button[melee.Button.BUTTON_Y]))
        Z_1.append(int(gamestate.players[p1].controller_state.button[melee.Button.BUTTON_Z]))
        fr.append(gamestate.frame)
        # c_stick_1_X.append(gamestate.players[p1].controller_state.c_stick[0])
        # c_stick_1_Y.append(gamestate.players[p1].controller_state.c_stick[1])
        c_direction = calculate_closest_m_direction(gamestate.players[p1].controller_state.c_stick[0],
                                                gamestate.players[p1].controller_state.c_stick[1])
        if c_direction == 'north':
            c_north_1.append(1.0)
        else:
            c_north_1.append(0.0)
        if c_direction == 'north_east':
            c_north_east_1.append(1.0)
        else:
            c_north_east_1.append(0.0)
        if c_direction == 'east':
            c_east_1.append(1.0)
        else:
            c_east_1.append(0.0)
        if c_direction == 'south_east':
            c_south_east_1.append(1.0)
        else:
            c_south_east_1.append(0.0)
        if c_direction == 'south':
            c_south_1.append(1.0)
        else:
            c_south_1.append(0.0)
        if c_direction == 'south_west':
            c_south_west_1.append(1.0)
        else:
            c_south_west_1.append(0.0)
        if c_direction == 'west':
            c_west_1.append(1.0)
        else:
            c_west_1.append(0.0)
        if c_direction == 'north_west':
            c_north_west_1.append(1.0)
        else:
            c_north_west_1.append(0.0)

        # main_stick_1_X.append(gamestate.players[p1].controller_state.main_stick[0])
        # main_stick_1_Y.append(gamestate.players[p1].controller_state.main_stick[1])
        m_direction = calculate_closest_m_direction(gamestate.players[p1].controller_state.main_stick[0],
                                                gamestate.players[p1].controller_state.main_stick[1])
        
        if m_direction == 'north':
            m_north_1.append(1.0)
        else:
            m_north_1.append(0.0)
        if m_direction == 'north_east':
            m_north_east_1.append(1.0)
        else:
            m_north_east_1.append(0.0)
        if m_direction == 'east':
            m_east_1.append(1.0)
        else:
            m_east_1.append(0.0)
        if m_direction == 'south_east':
            m_south_east_1.append(1.0)
        else:
            m_south_east_1.append(0.0)
        if m_direction == 'south':
            m_south_1.append(1.0)
        else:
            m_south_1.append(0.0)
        if m_direction == 'south_west':
            m_south_west_1.append(1.0)
        else:
            m_south_west_1.append(0.0)
        if m_direction == 'west':
            m_west_1.append(1.0)
        else:
            m_west_1.append(0.0)
        if m_direction == 'north_west':
            m_north_west_1.append(1.0)
        else:
            m_north_west_1.append(0.0)
        
        
        
        l_shoulder_1.append(round(gamestate.players[p1].controller_state.l_shoulder))
        r_shoulder_1.append(round(gamestate.players[p1].controller_state.r_shoulder))

        # Player 2 data
        A_2.append(int(gamestate.players[p2].controller_state.button[melee.Button.BUTTON_A]))
        B_2.append(int(gamestate.players[p2].controller_state.button[melee.Button.BUTTON_B]))
        X_2.append(int(gamestate.players[p2].controller_state.button[melee.Button.BUTTON_X]))
        Y_2.append(int(gamestate.players[p2].controller_state.button[melee.Button.BUTTON_Y]))
        Z_2.append(int(gamestate.players[p2].controller_state.button[melee.Button.BUTTON_Z]))
        
        # c_stick_2_X.append(gamestate.players[p2].controller_state.c_stick[0])
        # c_stick_2_Y.append(gamestate.players[p2].controller_state.c_stick[1])
        c_direction = calculate_closest_m_direction(gamestate.players[p1].controller_state.c_stick[0],
                                                gamestate.players[p1].controller_state.c_stick[1])
        if c_direction == 'north':
            c_north_2.append(1.0)
        else:
            c_north_2.append(0.0)
        if c_direction == 'north_east':
            c_north_east_2.append(1.0)
        else:
            c_north_east_2.append(0.0)
        if c_direction == 'east':
            c_east_2.append(1.0)
        else:
            c_east_2.append(0.0)
        if c_direction == 'south_east':
            c_south_east_2.append(1.0)
        else:
            c_south_east_2.append(0.0)
        if c_direction == 'south':
            c_south_2.append(1.0)
        else:
            c_south_2.append(0.0)
        if c_direction == 'south_west':
            c_south_west_2.append(1.0)
        else:
            c_south_west_2.append(0.0)
        if c_direction == 'west':
            c_west_2.append(1.0)
        else:
            c_west_2.append(0.0)
        if c_direction == 'north_west':
            c_north_west_2.append(1.0)
        else:
            c_north_west_2.append(0.0)

        # main_stick_2_X.append(gamestate.players[p2].controller_state.main_stick[0])
        # main_stick_2_Y.append(gamestate.players[p2].controller_state.main_stick[1])
        m_direction = calculate_closest_m_direction(gamestate.players[p1].controller_state.main_stick[0],
                                                gamestate.players[p1].controller_state.main_stick[1])
        if m_direction == 'north':
            m_north_2.append(1.0)
        else:
            m_north_2.append(0.0)
        if m_direction == 'north_east':
            m_north_east_2.append(1.0)
        else:
            m_north_east_2.append(0.0)
        if m_direction == 'east':
            m_east_2.append(1.0)
        else:
            m_east_2.append(0.0)
        if m_direction == 'south_east':
            m_south_east_2.append(1.0)
        else:
            m_south_east_2.append(0.0)
        if m_direction == 'south':
            m_south_2.append(1.0)
        else:
            m_south_2.append(0.0)
        if m_direction == 'south_west':
            m_south_west_2.append(1.0)
        else:
            m_south_west_2.append(0.0)
        if m_direction == 'west':
            m_west_2.append(1.0)
        else:
            m_west_2.append(0.0)
        if m_direction == 'north_west':
            m_north_west_2.append(1.0)
        else:
            m_north_west_2.append(0.0)
        l_shoulder_2.append(round(gamestate.players[p2].controller_state.l_shoulder))
        r_shoulder_2.append(round(gamestate.players[p2].controller_state.r_shoulder))
       
    #b = [A_1, A_2, B_1, B_2, X_1, X_2, Y_1, Y_2, Z_1, Z_2,m_north_1,m_north_east_1,m_east_1, m_south_east_1,m_south_east_1, c_stick_2_X, c_stick_2_Y ,main_stick_1_X,  main_stick_1_Y, main_stick_2_X, main_stick_2_Y, l_shoulder_1, l_shoulder_2, r_shoulder_1, r_shoulder_2]

    b = [A_1, A_2, B_1, B_2, X_1, X_2, Y_1, Y_2, Z_1, Z_2,
        m_north_1,m_north_east_1,m_east_1, m_south_east_1,m_south_1, m_south_west_1, m_west_1, m_north_west_1,
        m_north_2,m_north_east_2,m_east_2, m_south_east_2,m_south_2, m_south_west_2, m_west_2, m_north_west_2, 
        c_north_1,c_north_east_1,c_east_1, c_south_east_1,c_south_1, c_south_west_1, c_west_1, c_north_west_1,
        c_north_2,c_north_east_2,c_east_2, c_south_east_2,c_south_2, c_south_west_2, c_west_2, c_north_west_2, 
        l_shoulder_1, l_shoulder_2, r_shoulder_1, r_shoulder_2]
    columns = ['A_1', 'A_2', 'B_1', 'B_2', 'X_1', 'X_2', 'Y_1', 'Y_2', 'Z_1', 'Z_2', 
               'm_north_1','m_north_east_1','m_east_1', 'm_south_east_1','m_south_1', 'm_south_west_1', 'm_west_1', 'm_north_west_1',
                'm_north_2','m_north_east_2','m_east_2', 'm_south_east_2','m_south_2', 'm_south_west_2', 'm_west_2', 'm_north_west_2', 
                'c_north_1','c_north_east_1','c_east_1', 'c_south_east_1','c_south_1', 'c_south_west_1', 'c_west_1', 'c_north_west_1',
                'c_north_2','c_north_east_2','c_east_2', 'c_south_east_2','c_south_2', 'c_south_west_2', 'c_west_2', 'c_north_west_2', 
                'l_shoulder_1', 'l_shoulder_2', 'r_shoulder_1', 'r_shoulder_2']

    for ind, col in enumerate(columns):
        
        current_df[col] = b[ind]
    
    original_df = pd.concat([current_df, original_df])

    return original_df


def process_all_data(directory):
    filenames = os.listdir(directory)
    print(len(filenames))
    n = len(filenames)
    current_file = filenames[0]
    
    current_directory = directory + current_file
    original_df = pd.DataFrame(columns=range(46)) 
    #original_df.columns = ['A_1', 'A_2', 'B_1', 'B_2', 'X_1', 'X_2', 'Y_1', 'Y_2', 'Z_1', 'Z_2', 'c_stick_1_X',  'c_stick_1_Y', 'c_stick_2_X', 'c_stick_2_Y' ,'main_stick_1_X',  'main_stick_1_Y', 'main_stick_2_X','main_stick_2_Y', 'l_shoulder_1', 'l_shoulder_2', 'r_shoulder_1', 'r_shoulder_2']

    original_df.columns = ['A_1', 'A_2', 'B_1', 'B_2', 'X_1', 'X_2', 'Y_1', 'Y_2', 'Z_1', 'Z_2', 
               'm_north_1','m_north_east_1','m_east_1', 'm_south_east_1','m_south_1', 'm_south_west_1', 'm_west_1', 'm_north_west_1',
                'm_north_2','m_north_east_2','m_east_2', 'm_south_east_2','m_south_2', 'm_south_west_2', 'm_west_2', 'm_north_west_2', 
                'c_north_1','c_north_east_1','c_east_1', 'c_south_east_1','c_south_1', 'c_south_west_1', 'c_west_1', 'c_north_west_1',
                'c_north_2','c_north_east_2','c_east_2', 'c_south_east_2','c_south_2', 'c_south_west_2', 'c_west_2', 'c_north_west_2', 
                'l_shoulder_1', 'l_shoulder_2', 'r_shoulder_1', 'r_shoulder_2']
    counter = 0

    for i in range(0, 100):
        if i == 648:
            continue
        try:
            if check_game_for_fox(current_directory):
                counter +=1
                print(current_directory, '| Fox and FD |', i, " | ", counter)
                original_df = slippi_to_df(original_df, current_directory)
                # shutil.move('D:'+current_file, 'D:Data/'+current_file)
                current_file = filenames[i]
                current_directory = directory+current_file
            else:
                #print(current_file, '| Not fox and FD | ', i)
                current_file = filenames[i]
        except Exception as e:
            current_file = filenames[i]
            print(e)
        
    
    return original_df, counter




def Play():
    console = melee.Console(path=r"C:\Users\ivan1284\AppData\Roaming\Slippi Launcher\netplay", slippi_address="127.0.0.1", fullscreen=False)

    controller = melee.Controller(port=1, console=console)
    console.connect()
    console.run()
    controller.connect()
    columns = ['A_1', 'A_2', 'B_1', 'B_2', 'X_1', 'X_2', 'Y_1', 'Y_2', 'Z_1', 'Z_2', 
               'm_north_1','m_north_east_1','m_east_1', 'm_south_east_1','m_south_1', 'm_south_west_1', 'm_west_1', 'm_north_west_1',
                'm_north_2','m_north_east_2','m_east_2', 'm_south_east_2','m_south_2', 'm_south_west_2', 'm_west_2', 'm_north_west_2', 
                'c_north_1','c_north_east_1','c_east_1', 'c_south_east_1','c_south_1', 'c_south_west_1', 'c_west_1', 'c_north_west_1',
                'c_north_2','c_north_east_2','c_east_2', 'c_south_east_2','c_south_2', 'c_south_west_2', 'c_west_2', 'c_north_west_2', 
                'l_shoulder_1', 'l_shoulder_2', 'r_shoulder_1', 'r_shoulder_2']
    

    m = load_model('model100games.pkl')
    current_df = pd.DataFrame(columns=columns)



    while True:
        gamestate = console.step()
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            
          
            
            if gamestate.frame % 11 == 0:
                current_df = pd.DataFrame(columns=columns)
    
            else:
                
                Self, Opponent = gamestate.players.keys()
                input = generate_input(gamestate, Self, Opponent)

                current_df = pd.concat([current_df, input])


                if current_df.shape[0]>=10:
                    output = generate_out_RNN(m, current_df)
                    button, coordinates = interpret_output(output)
                    print(button)
                    if coordinates[0] != 10:
                        if button[0] == "c":
                            controller.tilt_analog(x=coordinates[0], y=coordinates[1], button=melee.Button.BUTTON_MAIN)
                            
                        else:
                            controller.simple_press(button=melee.Button.BUTTON_MAIN, x=coordinates[0], y=coordinates[1])
                    


                   


                    

                
            # print(np.array(current_sequence).shape)
            
            # print(m.predict(input))

                
        else:
            melee.menuhelper.MenuHelper.menu_helper_simple(gamestate,
                                                        controller,
                                                        melee.Character.FOX,
                                                        melee.Stage.FINAL_DESTINATION,
                                                        autostart=True,
                                                        swag=True)


def train_RNN_model(data:pd.DataFrame):
    seq_length = 10
    length = len(data)
    data = data.reset_index()
  
    train = data.iloc[:-seq_length]
    train =train.drop(columns="index")
    print(train.columns)
    targets = data.loc[seq_length:, ['A_1', 'B_1',  'X_1',  'Y_1', 'Z_1',
       'm_north_1', 'm_north_east_1', 'm_east_1', 'm_south_east_1',
       'm_south_1', 'm_south_west_1', 'm_west_1', 'm_north_west_1',
       'c_north_1', 'c_north_east_1', 'c_east_1', 'c_south_east_1',
       'c_south_1', 'c_south_west_1', 'c_west_1', 'c_north_west_1',
       'l_shoulder_1']]
    
    # targets = data.loc[seq_length]
    print(targets.shape)
    
    print('Created train, test, valid')
    np_train =np.asarray(train).astype(np.float16)
    np_targets = np.asarray(targets).astype(np.float16)
    train = train.apply(pd.to_numeric)

    
    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        np_train,
        targets=np_targets,
        batch_size=32,
        sequence_length=seq_length
    )
   
    
    mulvar_model = tf.keras.Sequential([
        

    tf.keras.layers.LSTM(75, input_shape=[None, 46], return_sequences=True, dropout=0.2, recurrent_dropout=0.2, activation="sigmoid"),
    tf.keras.layers.LSTM(46, return_sequences=True, activation="sigmoid",dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.LSTM(32, activation="sigmoid", dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(22, activation='softmax')
    ])
    loss = tf.keras.losses.CategoricalCrossentropy()
    optim = tf.keras.optimizers.Adam(learning_rate=0.00001)
    metrics = tf.keras.metrics.AUC(curve="PR")
    mulvar_model.compile(loss=loss, optimizer=optim, metrics=metrics)
    X = train.to_numpy()[np.newaxis, :1]
    print(X.shape)
    early_stop = EarlyStopping( monitor='loss', mode='min',
                           verbose=0.1, patience=10,
                           restore_best_weights=True)
    history = mulvar_model.fit(train_ds, epochs=5, callbacks=[early_stop])
    labels = ['A_1', 'B_1',  'X_1',  'Y_1', 'Z_1',
       'm_north_1', 'm_north_east_1', 'm_east_1', 'm_south_east_1',
       'm_south_1', 'm_south_west_1', 'm_west_1', 'm_north_west_1',
       'c_north_1', 'c_north_east_1', 'c_east_1', 'c_south_east_1',
       'c_south_1', 'c_south_west_1', 'c_west_1', 'c_north_west_1',
       'l_shoulder_1']
    Y_pred = mulvar_model.predict(X)

    # for i, val in enumerate(Y_pred[0]):
    #     print(labels[i],":", val)
    pickle_file_path = f'model100games.pkl'
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(mulvar_model, file)
    
    

def train_model(train, test, valid):
    X = train.drop(columns=['A_2',  'B_2', 'X_2',  'Y_2', 'Z_2',
       'c_stick_2_X', 'c_stick_2_Y',
       'main_stick_2_X', 'main_stick_2_Y', 
       'l_shoulder_2', 'r_shoulder_2'])
    Y = train.drop(columns=['A_1',  'B_1', 'X_1',  'Y_1', 'Z_1',
       'c_stick_1_X', 'c_stick_1_Y',
       'main_stick_1_X', 'main_stick_1_Y', 
       'l_shoulder_1', 'r_shoulder_1'])
    print(X.columns)
    print(Y.columns)
    X = np.asarray(X).astype(np.float32)
    Y = np.asarray(X).astype(np.float32)
    
    X_valid = valid.drop(columns=['A_2',  'B_2', 'X_2',  'Y_2', 'Z_2',
       'c_stick_2_X', 'c_stick_2_Y',
       'main_stick_2_X', 'main_stick_2_Y', 
       'l_shoulder_2', 'r_shoulder_2'])
    Y_valid = valid.drop(columns=['A_2',  'B_2', 'X_2',  'Y_2', 'Z_2',
       'c_stick_2_X', 'c_stick_2_Y',
       'main_stick_2_X', 'main_stick_2_Y', 
       'l_shoulder_2', 'r_shoulder_2'])
    X_valid = np.asarray(X).astype(np.float32)
    Y_valid = np.asarray(X).astype(np.float32)

    early_stop = EarlyStopping( monitor='val_loss', mode='min',
                           verbose=1, patience=10,
                           restore_best_weights=True)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='tanh', input_shape=(len(X[0]),)),
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dense(11, activation='softmax'),
    ])
    

    opt = tf.keras.optimizers.Adam(
        learning_rate=5e-5,
        name="Adam",
    )

    model.compile(
        optimizer=opt,
        loss='mean_squared_error',
        metrics=['accuracy']
    )
    model.fit(
        X,  # training data
        Y,  # training targets
        epochs=300,
        shuffle=True,
        callbacks=[early_stop],
        validation_data=(X_valid, Y_valid)
    )
    pickle_file_path = f'model100games.pkl'

    X_test = test.drop(columns=['A_2',  'B_2', 'X_2',  'Y_2', 'Z_2',
       'c_stick_2_X', 'c_stick_2_Y',
       'main_stick_2_X', 'main_stick_2_Y', 
       'l_shoulder_2', 'r_shoulder_2'])
    Y_test = test.drop(columns=['A_1',  'B_1', 'X_1',  'Y_1', 'Z_1',
       'c_stick_1_X', 'c_stick_1_Y',
       'main_stick_1_X', 'main_stick_1_Y', 
       'l_shoulder_1', 'r_shoulder_1'])
    labels = X_test.columns
    X_test = np.asarray(X).astype(np.float32)
    Y_test = np.asarray(X).astype(np.float32)
    
    a = model.predict(X_test)[30]
    
    for i in range(len(labels)):
        print(a[i], labels[i], Y_test[30][i])
    if not os.path.isdir('models'):
        os.mkdir(f'models/')

    with open(pickle_file_path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path):
    model = pickle.load(open(path, 'rb'))
    return model

def generate_input(gamestate: melee.GameState, Self ,Opponent):
    #self data
    c_north_1 = []
    c_north_east_1 = []
    c_east_1 = []
    c_south_east_1 = []
    c_south_1 = []
    c_south_west_1 = []
    c_west_1  = []
    c_north_west_1 = []
    A_1 = []
    B_1 = []
    X_1 = []
    Y_1 = []
    Z_1 = []
    m_north_1 = []
    m_north_east_1 = []
    m_east_1 = []
    m_south_east_1 = []
    m_south_1 = []
    m_south_west_1 = []
    m_west_1  = []
    m_north_west_1= []
    l_shoulder_1 = []
    r_shoulder_1 = []
    A_1.append(int(gamestate.players[Self].controller_state.button[melee.Button.BUTTON_A]))
    B_1.append(int(gamestate.players[Self].controller_state.button[melee.Button.BUTTON_B]))
    X_1.append(int(gamestate.players[Self].controller_state.button[melee.Button.BUTTON_X]))
    Y_1.append(int(gamestate.players[Self].controller_state.button[melee.Button.BUTTON_Y]))
    Z_1.append(int(gamestate.players[Self].controller_state.button[melee.Button.BUTTON_Z]))
    # c_stick_1_X.append(gamestate.players[p1].controller_state.c_stick[0])
    # c_stick_1_Y.append(gamestate.players[p1].controller_state.c_stick[1])
    c_direction = calculate_closest_m_direction(gamestate.players[Self].controller_state.c_stick[0],
                                            gamestate.players[Self].controller_state.c_stick[1])
    if c_direction == 'north':
        c_north_1.append(1.0)
    else:
        c_north_1.append(0.0)
    if c_direction == 'north_east':
        c_north_east_1.append(1.0)
    else:
        c_north_east_1.append(0.0)
    if c_direction == 'east':
        c_east_1.append(1.0)
    else:
        c_east_1.append(0.0)
    if c_direction == 'south_east':
        c_south_east_1.append(1.0)
    else:
        c_south_east_1.append(0.0)
    if c_direction == 'south':
        c_south_1.append(1.0)
    else:
        c_south_1.append(0.0)
    if c_direction == 'south_west':
        c_south_west_1.append(1.0)
    else:
        c_south_west_1.append(0.0)
    if c_direction == 'west':
        c_west_1.append(1.0)
    else:
        c_west_1.append(0.0)
    if c_direction == 'north_west':
        c_north_west_1.append(1.0)
    else:
        c_north_west_1.append(0.0)

    # main_stick_1_X.append(gamestate.players[p1].controller_state.main_stick[0])
    # main_stick_1_Y.append(gamestate.players[p1].controller_state.main_stick[1])
    m_direction = calculate_closest_m_direction(gamestate.players[Self].controller_state.main_stick[0],
                                            gamestate.players[Self].controller_state.main_stick[1])
    
    if m_direction == 'north':
        m_north_1.append(1.0)
    else:
        m_north_1.append(0.0)
    if m_direction == 'north_east':
        m_north_east_1.append(1.0)
    else:
        m_north_east_1.append(0.0)
    if m_direction == 'east':
        m_east_1.append(1.0)
    else:
        m_east_1.append(0.0)
    if m_direction == 'south_east':
        m_south_east_1.append(1.0)
    else:
        m_south_east_1.append(0.0)
    if m_direction == 'south':
        m_south_1.append(1.0)
    else:
        m_south_1.append(0.0)
    if m_direction == 'south_west':
        m_south_west_1.append(1.0)
    else:
        m_south_west_1.append(0.0)
    if m_direction == 'west':
        m_west_1.append(1.0)
    else:
        m_west_1.append(0.0)
    if m_direction == 'north_west':
        m_north_west_1.append(1.0)
    else:
        m_north_west_1.append(0.0)
    
    
    
    l_shoulder_1.append(round(gamestate.players[Self].controller_state.l_shoulder))
    r_shoulder_1.append(round(gamestate.players[Self].controller_state.r_shoulder))



    #opponent data
    c_north_2 = []
    c_north_east_2 = []
    c_east_2 = []
    c_south_east_2 = []
    c_south_2 = []
    c_south_west_2 = []
    c_west_2  = []
    c_north_west_2 = []
    A_2 = []
    B_2 = []
    X_2 = []
    Y_2 = []
    Z_2 = []
    main_stick_2_X = []
    main_stick_2_Y = []
    m_north_2 = []
    m_north_east_2 = []
    m_east_2 = []
    m_south_east_2 = []
    m_south_2 = []
    m_south_west_2 = []
    m_west_2  = []
    m_north_west_2 = []
    l_shoulder_2 = []
    r_shoulder_2 = []
    A_2.append(int(gamestate.players[Opponent].controller_state.button[melee.Button.BUTTON_A]))
    B_2.append(int(gamestate.players[Opponent].controller_state.button[melee.Button.BUTTON_B]))
    X_2.append(int(gamestate.players[Opponent].controller_state.button[melee.Button.BUTTON_X]))
    Y_2.append(int(gamestate.players[Opponent].controller_state.button[melee.Button.BUTTON_Y]))
    Z_2.append(int(gamestate.players[Opponent].controller_state.button[melee.Button.BUTTON_Z]))
    
    # c_stick_2_X.append(gamestate.players[p2].controller_state.c_stick[0])
    # c_stick_2_Y.append(gamestate.players[p2].controller_state.c_stick[1])
    c_direction = calculate_closest_m_direction(gamestate.players[Opponent].controller_state.c_stick[0],
                                            gamestate.players[Opponent].controller_state.c_stick[1])
    if c_direction == 'north':
        c_north_2.append(1.0)
    else:
        c_north_2.append(0.0)
    if c_direction == 'north_east':
        c_north_east_2.append(1.0)
    else:
        c_north_east_2.append(0.0)
    if c_direction == 'east':
        c_east_2.append(1.0)
    else:
        c_east_2.append(0.0)
    if c_direction == 'south_east':
        c_south_east_2.append(1.0)
    else:
        c_south_east_2.append(0.0)
    if c_direction == 'south':
        c_south_2.append(1.0)
    else:
        c_south_2.append(0.0)
    if c_direction == 'south_west':
        c_south_west_2.append(1.0)
    else:
        c_south_west_2.append(0.0)
    if c_direction == 'west':
        c_west_2.append(1.0)
    else:
        c_west_2.append(0.0)
    if c_direction == 'north_west':
        c_north_west_2.append(1.0)
    else:
        c_north_west_2.append(0.0)

    # main_stick_2_X.append(gamestate.players[p2].controller_state.main_stick[0])
    # main_stick_2_Y.append(gamestate.players[p2].controller_state.main_stick[1])
    m_direction = calculate_closest_m_direction(gamestate.players[Opponent].controller_state.main_stick[0],
                                            gamestate.players[Opponent].controller_state.main_stick[1])
    if m_direction == 'north':
        m_north_2.append(1.0)
    else:
        m_north_2.append(0.0)
    if m_direction == 'north_east':
        m_north_east_2.append(1.0)
    else:
        m_north_east_2.append(0.0)
    if m_direction == 'east':
        m_east_2.append(1.0)
    else:
        m_east_2.append(0.0)
    if m_direction == 'south_east':
        m_south_east_2.append(1.0)
    else:
        m_south_east_2.append(0.0)
    if m_direction == 'south':
        m_south_2.append(1.0)
    else:
        m_south_2.append(0.0)
    if m_direction == 'south_west':
        m_south_west_2.append(1.0)
    else:
        m_south_west_2.append(0.0)
    if m_direction == 'west':
        m_west_2.append(1.0)
    else:
        m_west_2.append(0.0)
    if m_direction == 'north_west':
        m_north_west_2.append(1.0)
    else:
        m_north_west_2.append(0.0)
    l_shoulder_2.append(round(gamestate.players[Opponent].controller_state.l_shoulder))
    r_shoulder_2.append(round(gamestate.players[Opponent].controller_state.r_shoulder))
    b = [A_1, A_2, B_1, B_2, X_1, X_2, Y_1, Y_2, Z_1, Z_2,
        m_north_1,m_north_east_1,m_east_1, m_south_east_1,m_south_1, m_south_west_1, m_west_1, m_north_west_1,
        m_north_2,m_north_east_2,m_east_2, m_south_east_2,m_south_2, m_south_west_2, m_west_2, m_north_west_2, 
        c_north_1,c_north_east_1,c_east_1, c_south_east_1,c_south_1, c_south_west_1, c_west_1, c_north_west_1,
        c_north_2,c_north_east_2,c_east_2, c_south_east_2,c_south_2, c_south_west_2, c_west_2, c_north_west_2, 
        l_shoulder_1, l_shoulder_2, r_shoulder_1, r_shoulder_2]
    columns = ['A_1', 'A_2', 'B_1', 'B_2', 'X_1', 'X_2', 'Y_1', 'Y_2', 'Z_1', 'Z_2', 
               'm_north_1','m_north_east_1','m_east_1', 'm_south_east_1','m_south_1', 'm_south_west_1', 'm_west_1', 'm_north_west_1',
                'm_north_2','m_north_east_2','m_east_2', 'm_south_east_2','m_south_2', 'm_south_west_2', 'm_west_2', 'm_north_west_2', 
                'c_north_1','c_north_east_1','c_east_1', 'c_south_east_1','c_south_1', 'c_south_west_1', 'c_west_1', 'c_north_west_1',
                'c_north_2','c_north_east_2','c_east_2', 'c_south_east_2','c_south_2', 'c_south_west_2', 'c_west_2', 'c_north_west_2', 
                'l_shoulder_1', 'l_shoulder_2', 'r_shoulder_1', 'r_shoulder_2']
   
    current_df = pd.DataFrame(columns=range(46))
    current_df.columns = columns
    
    for ind, col in enumerate(columns):
        current_df[col] = b[ind]
    return current_df

def generate_out_RNN(model, input):
    labels = ['A_1', 'B_1',  'X_1',  'Y_1', 'Z_1',
       'm_north_1', 'm_north_east_1', 'm_east_1', 'm_south_east_1',
       'm_south_1', 'm_south_west_1', 'm_west_1', 'm_north_west_1',
       'c_north_1', 'c_north_east_1', 'c_east_1', 'c_south_east_1',
       'c_south_1', 'c_south_west_1', 'c_west_1', 'c_north_west_1',
       'l_shoulder_1']
    input = input.apply(pd.to_numeric)
    X = input.to_numpy()[np.newaxis, :10]
    prediction = (model.predict(X))
    
    predicted = {}
    for i, val in enumerate(prediction[0]):
        
        predicted[labels[i]] = val
    print(predicted)
    return predicted


def interpret_output(prediction: dict):
    button = max(prediction, key=prediction.get)

    if len(button)>=7 and button[0] != "l":
        direction = button[2:-2]
        print(direction)
        m_directions = {'north':[0.5,1],'north_east':[0.75,0.75], 'east':[1.0, 0.5],
                  'south_east':[0.75, 0.35], 'south':[0.5, 0], 'south_west':[0.35, 0.35],
                  'west':[0, 0.5], 'north_west': [0.35, 0.75]}
        return button, m_directions[direction]
    else:
        return button, [10,10]
    



def generate_output(model, input):
    prediction = model.predict(input)
    print(prediction)
    labels = ['A_1',  'B_1', 'X_1',  'Y_1', 'Z_1', 'c_stick_1_X', 'c_stick_1_Y','main_stick_1_X', 'main_stick_1_Y', 'l_shoulder_1', 'r_shoulder_1']
    predicted = {}
    for ind, val in enumerate(prediction[0]):
        if val<0.06:
            val += 0.25
            predicted[labels[ind]]=val
        else:
            if labels[ind] == 'main_stick_1_X' or labels[ind] == 'main_stick_1_Y' or labels[ind]=='c_stick_1_X' or labels[ind]=='c_stick_1_Y':
                val+=0.25
                
                
                predicted[labels[ind]]=val
    return predicted
    
    
def interpret_out(predicted):
    mapping_of_buttons = {'A_1': melee.Button.BUTTON_A, 'B_1': melee.Button.BUTTON_B, 'X_1':melee.Button.BUTTON_Y, 'Y_1': melee.Button.BUTTON_Y,
                          'Z_1': melee.Button.BUTTON_Z, 'c_stick_1_X':'Xc', 'c_stick_1_Y':'Yc', 'main_stick_1_X':'Xm', 'main_stick_1_Y':'Ym',
                          'l_shoulder_1':'shield', 'r_shoulder_1':'shield'}
    guessing_max = ''
    value_of_pred = -1
    print(predicted)
    for i in predicted.keys():
        if predicted[i] > value_of_pred:
            guessing_max = i
            value_of_pred = predicted[i]
    if guessing_max == 'c_stick_1_X':
        return {mapping_of_buttons[guessing_max]:value_of_pred, mapping_of_buttons["c_stick_1_Y"]:predicted['c_stick_1_Y']}
    if guessing_max == 'c_stick_1_Y':
        return {mapping_of_buttons['c_stick_1_X']:predicted['c_stick_1_X'], mapping_of_buttons[guessing_max]:value_of_pred}
    if guessing_max == 'main_stick_1_X':
        return {mapping_of_buttons[guessing_max]:value_of_pred, mapping_of_buttons['main_stick_1_Y']:predicted['main_stick_1_Y']}
    if guessing_max == 'main_stick_1_Y':
        return {mapping_of_buttons['main_stick_1_X']:predicted['main_stick_1_X'],mapping_of_buttons[guessing_max]:value_of_pred}
    return {mapping_of_buttons[guessing_max]:1}


def main():
    

    # original_df = pd.DataFrame() 
    # df, count = process_all_data('D:/Data/')
    # df.to_pickle(file_name)
    # print(count)
    # df = slippi_to_df(original_df, 'UnevenDefensiveGoldfish.slp')
    # print(df)

    df = pd.read_pickle('D:AllFoxFalcoAndFD_X_Y_1_hot100.pkl')
    print(df.columns)
  
    
   
    #train_RNN_model(df)
    Play()

    












    # print('Read pickle')
    # train = df.iloc[0:len(df)//2]
    # print(type(train))
    # test = df.iloc[len(df)//2:len(df)//2+len(df)//4]
    # valid = df.iloc[(len(df)//4 )*3:]
    # print('Created train, test, valid')
    # train_d = np.asarray(train).astype(np.float32)
    # train_model(train=train, test=test, valid=valid)
    # Play()

    
    
    
   

    
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
 