# Rice_leaf_disease_detection

-- This is my Computer Vision assignment/project about > rice leaf disease recognition using deep learning < 

-- You can clone this project and follow some of the instructions below to get this to operate
1. Clone this repo in Git Bash using   >>> git clone <this-git-repo-url>   command
2. Open Visual Studio Code, import the installed folders, and then open the terminal
3. Copy-paste this  >>> pip install numpy pandas matplotlib tensorflow scikit-learn pillow opencv-python
4. Reinstall the venv framework by copying and pasting the following command:   >>> python -m venv venv


-- And here are the project components to help you understand the project's workflow
-- The modules are
    > data                       # where you put the datasets (images) of the rice leaf 
    > model                      # the brain of the program, it accumulates knowledge learnt from the datasets
    > rice_leaf                  # all the main codes are in here 
    app.py                       # where you run the whole program ( >>> streamlit run app.py )
    auth.py                      # create users' info and password as "key-value" structure via hash process  
    class_indices.json           # categorize the sub-folders in data folder as json objects
    main.py                      # where you run this project as a prototype (demo)
    README.md                    # for noting during work
    requirements.txt             # list of important Python libraries ( must install )
    save.txt                     # just a draft paper ( skip this one )
    users.json                   # where the system stores and deletes users' info 
    > venv framework             # python virtual environment ( must install )


-- Okay, now you have completed the basic setup and understand how the program works
-- Now let me show you how to use the program properly
1. Put the datasets package inside the (data) folder/ subfolders ( infected rice leaf images )
2. Open the terminal and type    >>> npm run train.py
3. Check if the (model) folder has a file with the .h5 at the end
4. Run     >>> npm run main.py ( demo )
5. Run     >>> streamlit run app.py
6. Fill the register/login form and check ( users.json ) for the user's information generated
