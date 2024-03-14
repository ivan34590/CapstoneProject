import melee
import pandas as pd
import os
import time
import re
import numpy as np



        

def contains_fox_in_file(filename):
    return ('Fox' in filename and 'FD' in filename) or ('Fox' in filename and 'Final Destination' in filename)

def check_game_for_fox(filename):
    console = melee.Console(path=filename, system="file", allow_old_version=True)
    console.connect()
    gamestate=console.step()
    p1, p2 = gamestate.players.keys()
    return (gamestate.players[p1].character == melee.Character.FOX or gamestate.players[p2].character == melee.Character.FOX) and (gamestate.stage == melee.Stage.FINAL_DESTINATION)


def slippi_to_df(original_df, filename):
    console = melee.Console(path=filename, system="file", allow_old_version=True)
    current_df = pd.DataFrame()
    console.connect()
    fr = []


    #Player 1 data
    c_stick_1_X = []
    c_stick_1_Y = []
    A_1 = []
    B_1 = []
    X_1 = []
    Y_1 = []
    Z_1 = []
    main_stick_1_X = []
    main_stick_1_Y = []
    l_shoulder_1 = []
    r_shoulder_1 = []

    #Player 2 data

    c_stick_2_X = []
    c_stick_2_Y = []
    A_2 = []
    B_2 = []
    X_2 = []
    Y_2 = []
    Z_2 = []
    main_stick_2_X = []
    main_stick_2_Y = []
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
        c_stick_1_X.append(gamestate.players[p1].controller_state.c_stick[0])
        c_stick_1_Y.append(gamestate.players[p1].controller_state.c_stick[1])

        main_stick_1_X.append(gamestate.players[p1].controller_state.main_stick[0])
        main_stick_1_Y.append(gamestate.players[p1].controller_state.main_stick[1])
        l_shoulder_1.append(gamestate.players[p1].controller_state.l_shoulder)
        r_shoulder_1.append(gamestate.players[p1].controller_state.r_shoulder)

        # Player 2 data
        A_2.append(int(gamestate.players[p2].controller_state.button[melee.Button.BUTTON_A]))
        B_2.append(int(gamestate.players[p2].controller_state.button[melee.Button.BUTTON_B]))
        X_2.append(int(gamestate.players[p2].controller_state.button[melee.Button.BUTTON_X]))
        Y_2.append(int(gamestate.players[p2].controller_state.button[melee.Button.BUTTON_Y]))
        Z_2.append(int(gamestate.players[p2].controller_state.button[melee.Button.BUTTON_Z]))
        
        c_stick_2_X.append(gamestate.players[p2].controller_state.c_stick[0])
        c_stick_2_Y.append(gamestate.players[p2].controller_state.c_stick[1])

        main_stick_2_X.append(gamestate.players[p2].controller_state.main_stick[0])
        main_stick_2_Y.append(gamestate.players[p2].controller_state.main_stick[1])
        l_shoulder_2.append(gamestate.players[p2].controller_state.l_shoulder)
        r_shoulder_2.append(gamestate.players[p2].controller_state.r_shoulder)
       
        
    b = [A_1, A_2, B_1, B_2, X_1, X_2, Y_1, Y_2, Z_1, Z_2, c_stick_1_X,  c_stick_1_Y, c_stick_2_X, c_stick_2_Y ,main_stick_1_X,  main_stick_1_Y, main_stick_2_X, main_stick_2_Y, l_shoulder_1, l_shoulder_2, r_shoulder_1, r_shoulder_2]
    columns = ['A_1', 'A_2', 'B_1', 'B_2', 'X_1', 'X_2', 'Y_1', 'Y_2', 'Z_1', 'Z_2', 'c_stick_1_X',  'c_stick_1_Y', 'c_stick_2_X', 'c_stick_2_Y' ,'main_stick_1_X',  'main_stick_1_Y', 'main_stick_2_X', 'main_stick_2_Y', 'l_shoulder_1', 'l_shoulder_2', 'r_shoulder_1', 'r_shoulder_2']

    for ind, col in enumerate(columns):
        current_df[col] = b[ind]
    
    original_df = pd.concat([current_df, original_df])

    return original_df


def process_all_data(directory):
    filenames = os.listdir(directory)
    print(len(filenames))
    n = len(filenames)
    current_file = filenames[70000]
    
    current_directory = directory + current_file
    original_df = pd.DataFrame(columns=range(22)) 
    original_df.columns = ['A_1', 'A_2', 'B_1', 'B_2', 'X_1', 'X_2', 'Y_1', 'Y_2', 'Z_1', 'Z_2', 'c_stick_1_X',  'c_stick_1_Y', 'c_stick_2_X', 'c_stick_2_Y' ,'main_stick_1_X',  'main_stick_1_Y', 'main_stick_2_X','main_stick_2_Y', 'l_shoulder_1', 'l_shoulder_2', 'r_shoulder_1', 'r_shoulder_2']
    counter = 0
    #contains_fox_in_file(current_file)
    for i in range(70000, 80000):
        
        try:
            if contains_fox_in_file(current_file) or check_game_for_fox('D:'+current_file):
                counter +=1
                print(current_file, '| Fox and FD |', i, " | ", counter)
                original_df = slippi_to_df(original_df, current_directory)
                current_file = filenames[i]
                current_directory = directory+current_file
            else:
                #print(current_file, '| Not fox and FD | ', i)
                current_file = filenames[i]
        except Exception:
            current_file = filenames[i]
            print('Not a slp file')
        
    
    return original_df

# testing not relevant        
# def doRandomstuff():
#     return melee.Button.BUTTON_A

def Play():
    console = melee.Console(path=r"C:\Users\ivan1284\AppData\Roaming\Slippi Launcher\netplay", slippi_address="127.0.0.1", fullscreen=False)

    controller = melee.Controller(port=1, console=console)
    console.connect()
    console.run()
    controller.connect()
    while True:
        gamestate = console.step()
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                if gamestate.frame %2==0:

                    controller.press_button(melee.Button.BUTTON_B)
                    print(gamestate.players[1].controller_state.button[melee.Button.BUTTON_B])
                else:
                    controller.release_all()
                    print(gamestate.players[1].controller_state.button[melee.Button.BUTTON_B])

                
        else:
            melee.menuhelper.MenuHelper.menu_helper_simple(gamestate,
                                                        controller,
                                                        melee.Character.FOX,
                                                        melee.Stage.FINAL_DESTINATION,
                                                        autostart=True,
                                                        swag=True)


def main():
    # df1 = pd.DataFrame()
    # df = check_game_for_fox('D:22_17_36 Fox + [SUR] Marth (FD).slp')
    # print(df)
    # Play()
    df = process_all_data(directory='D:')
    # print(contains_fox_in_file('00_39_58 Falco + [TITP] Fox (YI).slp'))
    print(df.shape)
    df.to_pickle('D:AllFoxAndFD70000-80000.pkl')

    
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
 