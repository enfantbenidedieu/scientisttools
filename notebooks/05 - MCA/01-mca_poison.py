#-------------------------------------------------------------------------------------
##
#
import warnings
#from itables import init_notebook_mode
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

#load poison dataset
from scientisttools import load_poison
poison = load_poison()
poison.head(6)

from scientisttools import MCA
res_mca = MCA(n_components=5,ind_sup=[50,51,52,53,54],quali_sup = [2,3],quanti_sup =[0,1])
res_mca.fit(poison)

