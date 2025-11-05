import numpy as np
import time
import pyautogui
    
def click_icone(path,boucle=10,wait=0.3,gris=True,confidence=0.95):
    while boucle > 0:
        time.sleep(np.random.uniform(wait, wait*3)) 
        find = pyautogui.locateOnScreen(path, grayscale=gris, confidence=confidence)
        if find is not None:
            find = pyautogui.center(find)
            pyautogui.leftClick(find,duration=0.3)
            return True
        boucle -=1
    print(f'Looking For {path[7:]}')
    return False

def convert_to_float(s):
    """
    Converts a string representing a number into a float by inserting a decimal point
    before the last two digits if it doesn't already contain one.
    
    Parameters:
    s (str): The input string representing the number.
    
    Returns:
    float: The converted float value.
    """
    if not s or not isinstance(s, str):
        return None  # Return None if s is None or not a string

    # Remove any whitespace and replace comma with period
    s = s.strip().replace(',', '.')

    # Remove unwanted characters, keeping only digits and periods
    s_clean = ''.join(c for c in s if c.isdigit() or c == '.')

    if not s_clean or s_clean == '.':
        return None  # Return None if s_clean is empty or just a peri
    
    # Keep only digit characters
    s_digits = ''.join(filter(str.isdigit, s))
    
    # Ensure there are at least two digits
    s_digits = s_digits.zfill(2)
    
    # Insert decimal point before the last two digits
    s_new = s_digits[:-2] + '.' + s_digits[-2:]
    
    try:
        return float(s_new)
    except ValueError:
        raise ValueError(f"Invalid number format after processing: {s_new}")
        return None 
