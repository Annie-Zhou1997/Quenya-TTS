from Preprocessing.qya_g2p import process_complete_sentence as Quenya_G2P
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
text = "Et Eärello Endorenna utúlien. Sinome maruvan ar Hildinyar, tenn’ Ambar-metta!"
# print(Quenya_G2P(text))

tf = ArticulatoryCombinedTextFrontend(language="qya")
tf.string_to_tensor(text, view=True)

tf = ArticulatoryCombinedTextFrontend(language="en")
tf.string_to_tensor("hello world", view=True)

