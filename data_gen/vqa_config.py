open_close_status = {
    "Toilet": ["lid", "seat"],
    "Door": ["surface_board", "rotation_door"],
    "Laptop": ["shaft"],
    "StorageFurniture": ["cabinet_door", "door", "drawer"],
    "Table": ["drawer"],
    "Window": ["rotation", "translation"],
    "TrashCan": ["lid"],
    "USB": ["cap"],
    #   "KitchenPot": ["lid"], # too small change
    "Refrigerator": ["door", "other_leaf"],
    #   "Bucket": ["lid"],
    #   "CoffeeMachine": ["lid", "portafilter"],
    "Microwave": ["door"],
    "Oven": ["door"],
    #   "Bottle": ["lid"],
    #   "Lighter": ["lid"],
    #   "Camera": ["lid"],
    "Dishwasher": ["door"],
    "Pen": ["cap"],
    "Safe": ["door"],
    #   "Printer": ["lid", "drawer"],
    "WashingMachine": ["door"],
    "Box": ["rotation_lid"],
    "Stapler": ["lid"],
    "Suitcase": ["lid"],
    "Phone": ["flipping_lid", "rotation_lid", "slider"],
}

action_primtives = ["slide_open", "slide_close", "flap_open", "flap_close", "cap", "uncap", "pick", "place", "slide_in", "slide_out", "wipe", "press", "rotate", "StatusComplete"]

# 2D task instructions
DET_ALL_ROT_INSTRUCT = "Detect all manipulable object parts and provide their 2D rotated bounding boxes."
DET_ALL_INSTRUCT = "Detect all manipulable object parts and provide their 2D bounding boxes."

REC_JOINT_ROT_INSTRUCT = "Please provide the joint's type and its 2D rotated bounding box linked to the object part {REF}."
REC_JOINT_ROT_EXT_INSTRUCT = "Please provide the joint's type and its 2D rotated bounding box with depth linked to the object part {REF}."
REG_STATUS_INSTRUCT = "What is the status of the object part {REF}?"
REC_SINGLE_LINK_INSTRUCT = "Please provide the 2D rotated bounding box of the region this sentence describes: "
GROUNDING_ACTIONS_INSTRUCT = "Please execute the task described wih 2D rotated bounding box representations by the following instruction: "

# 3D task instructions
DET_ALL_BBOX_3D_INSTRUCT = "Detect all manipulable object parts and provide their 3D bounding boxes."
DET_ALL_3D_INSTRUCT = "Detect all manipulable object parts and provide their 3D bounding boxes."

REC_JOINT_3D_INSTRUCT = "Please provide the joint's type and its 3D axis linked to the object part {REF}."
REG_STATUS_3D_INSTRUCT = "What is the status of the object part {REF}?"
REC_SINGLE_LINK_3D_INSTRUCT = "Please provide the 3D bounding box of the region this sentence describes: "
GROUNDING_ACTIONS_3D_INSTRUCT = "Please execute the task described wih 3D rotated bounding box representations by the following instruction: "
DET_AFFORDANCE_3D_INSTRUCT = "Please provide the 3D bounding box of the region where the action could be applied: "

# box delimiters
DELIMIMTER_ROTATED_BOX_START = "<rb>"
DELIMIMTER_ROTATED_BOX_END = "</rb>"

DELIMIMTER_ROTATED_BOX_DEPTH_START = "<rbd>"
DELIMIMTER_ROTATED_BOX_DEPTH_END = "</rbd>"

DELIMIMTER_DEPTH_START = "<dep>"
DELIMIMTER_DEPTH_END = "</dep>"

DELIMIMTER_BOX_START = "<p>"
DELIMIMTER_BOX_END = "</p>"
DELIMIMTER_BOX_3D_START = "<box>"
DELIMIMTER_BOX_3D_END = "</box>"
DELIMIMTER_AXIS_3D_START = "<axis>"
DELIMIMTER_AXIS_3D_END = "</axis>"

joint_types_mapping = {
    "free": "continuous",
    "heavy": "fixed",
    "hinge": "revolute",
    "slider": "prismatic",
    "slider+": "prismatic",
    "static": "fixed",
}

NONE_PLACEHOLDER = -10000
DET_ALL_SKIPPED_CLASS = ["Keyboard", "Phone", "Remote"]
HOLDOUT_CLASSES = ["Toilet", "USB", "Scissors", "Stapler", "Kettle", "Oven", "Phone", "WashingMachine"]
