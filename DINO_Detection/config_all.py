label_2_id = {
  "Carpetweed": 4,
  "PalmerAmaranth": 8,
  "Eclipta": 6,
  "Purslane": 2,
  "Waterhemp": 0,
  "SpottedSpurge": 3,
  "MorningGlory": 1,
  "Goosegrass": 10,
  "PricklySida": 7,
  "Sicklepod": 9,
  "CutleafGroundcherry": 11,
  "Ragweed": 5
}
# label_2_id = {
#     "Unblue_visible": 0,
#     "Blue_occluded": 1,
#     "Unblue_occluded": 2,
#     "Blue_visible": 3
# }

id2name = {}
id2name = {
    0:'Waterhemp',
    1:"MorningGlory",
    2:"Purslane",
    3:"SpottedSpurge",
    4:"Carpetweed",
    5:"Ragweed",
    6:"Eclipta",
    7:"PricklySida",
    8:"PalmerAmaranth",
    9:"Sicklepod",
    10:"Goosegrass",
    11:"CutleafGroundcherry",
}
# id2name = {0: "Unblue_visible", 1: "Blue_occluded", 2: "Unblue_occluded", 3: "Blue_visible"}

# blueberry
# maxDet = [1, 900, 3000]
maxDet = None
dn_number_default = 100
# dn_number_init = 3000
dn_number_init = dn_number_default
# dn_labelbook_size = 1024
dn_labelbook_size = 100