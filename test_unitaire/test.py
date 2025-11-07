# 2 boutons + new fichier

import threading
import time
import PySimpleGUI as sg
from objet import controller

def update_table(window, controller):
    multiline = window['-MULTILINE-']
    multiline.update(value=controller)

layout = [
    [sg.Button('Lancer le scan', key='-SCAN-')],
    [sg.Button('Draw', key='-DRAW-')],  # Add this line for the Draw button
    [sg.Multiline(size=(80, 20), key='-MULTILINE-')],  # Replace Table with Multiline
    [sg.Button('Quitter')]
]

window = sg.Window('Interface Graphique', layout, location=(3245,140), size=(1200, 800), finalize=True)

controller = controller.Controller()
scan_thread = None

def scan_thread_function():
    while controller.running:
        texte_table = controller.main()
        update_table(window, texte_table)
        window.refresh()  # Update GUI
        time.sleep(0.5)  # Wait 0.5 second

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == "Quitter":
        controller.running = False  # Stop the scan thread
        break
    elif event == "-SCAN-":
        if not controller.running:
            controller.running = True
            scan_thread = threading.Thread(target=scan_thread_function)
            scan_thread.start()  # Start the scan thread if not already done
    elif event == "-DRAW-":  # Handle the Draw button event
        controller.draw()  # Call the draw method from the controller
window.close()


