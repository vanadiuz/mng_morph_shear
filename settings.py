 
#    _______   ___        ______    _______       __      ___           ___      ___  __        _______    __          __       _______   ___       _______   ________  
#   /" _   "| |"  |      /    " \  |   _  "\     /""\    |"  |         |"  \    /"  |/""\      /"      \  |" \        /""\     |   _  "\ |"  |     /"     "| /"       ) 
#  (: ( \___) ||  |     // ____  \ (. |_)  :)   /    \   ||  |          \   \  //  //    \    |:        | ||  |      /    \    (. |_)  :)||  |    (: ______)(:   \___/  
#   \/ \      |:  |    /  /    ) :)|:     \/   /' /\  \  |:  |           \\  \/. .//' /\  \   |_____/   ) |:  |     /' /\  \   |:     \/ |:  |     \/    |   \___  \    
#$ //  \ ___  \  |___(: (____/ // (|  _  \\  //  __'  \  \  |___         \.    ////  __'  \   //      /  |.  |    //  __'  \  (|  _  \\  \  |___  // ___)_   __/  \\   
#  (:   _(  _|( \_|:  \\        /  |: |_)  :)/   /  \\  \( \_|:  \         \\   //   /  \\  \ |:  __   \  /\  |\  /   /  \\  \ |: |_)  :)( \_|:  \(:      "| /" \   :)  
#   \_______)  \_______)\"_____/   (_______/(___/    \___)\_______)         \__/(___/    \___)|__|  \___)(__\_|_)(___/    \___)(_______/  \_______)\_______)(_______/   

import os                                                                                                                                                    
 
def init():

    global home
    global global_path #data is here
    global script_path

    home = '/home/lv70806/Vania/' #os.environ["HOME"]
    global_path = "/binfl/lv70806/Vania/" 
    script_path = os.path.dirname(os.path.realpath(__file__)) + '/'
