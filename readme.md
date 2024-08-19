# A complete automated script for User authentication preventing Spoofing Attacks.
**Preface :** Normal deep learning techniques and facial recognition systems are susceptible for spoofing attacks by paper-cut photos or videos.

1) This is a deep learning framework designed to improve the authenticty of the existing deep learning facial recognition systems with the help of depth maps to distinguish between live and a spoof.

**Usage :** \\
    1) Generate facial embeddings of users.

    `python authenticate.py --generate '<path to images>'`   

    Note : <Path to images> should be replaced with the path of the folder containing images of the user. Make sure that the path is within quotes ' ' or " " .

    2) Authenticate users
    `python authenticate.py`  

