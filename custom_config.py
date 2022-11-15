labels_to_names = {
    0: 'unknown',
    1: 'off',
    2: 'red',
    3: 'yellow',
    4: 'green',
    5: 'redRight',
    6: 'yellowRight',
    7: 'greenRight',
    8: 'redLeft',
    9: 'yellowLeft',
    10: 'greenLeft',
    11: 'red,greenRight',
    12: 'yellow,greenRight',
    13: 'green,greenRight',
    14: 'red,yellowRight',
    15: 'yellow,yellowRight',
    16: 'green,yellowRight',
    17: 'red,redRight',
    18: 'yellow,redRight',
    19: 'green,redRight',
    20: 'red,greenLeft',
    21: 'yellow,greenLeft',
    22: 'green,greenLeft'
}


pipeline_labels_new_labels = {
    "red":
        {
            "circle":  2,
            "left_arrow": 8,
            "right_arrow": 5,
            "up_arrow": 0,
            "down_arrow": 0,
            "down_left_arrow": 0,
            "down_right_arrow": 0,
            "cross": 0,
        },
    "amber": {
            "circle": 3,
            "left_arrow": 9,
            "right_arrow": 6,
            "up_arrow": 0,
            "down_arrow": 0,
            "down_left_arrow": 0,
            "down_right_arrow": 0,
            "cross": 0,
            },
    "green": {
            "circle": 4,
            "left_arrow": 10,
            "right_arrow": 7,
            "up_arrow": 0,
            "down_arrow": 0,
            "down_left_arrow": 0,
            "down_right_arrow": 0,
            "cross": 0,
            },
    "white":
        {
            "circle": 0,
            "left_arrow": 0,
            "right_arrow": 0,
            "up_arrow": 0,
            "down_arrow": 0,
            "down_left_arrow": 0,
            "down_right_arrow": 0,
            "cross": 0,
            },
    "unknown": {"unknown": 0}
}
