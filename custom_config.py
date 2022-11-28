labels_to_names = {
    0: 'unknown',
    1: 'red',
    2: 'yellow',
    3: 'green',
    4: 'redRight',
    5: 'yellowRight',
    6: 'greenRight',
    7: 'redLeft',
    8: 'yellowLeft',
    9: 'greenLeft',
    10: 'red,greenRight',
    11: 'yellow,greenRight',
    12: 'green,greenRight',
    13: 'red,yellowRight',
    14: 'yellow,yellowRight',
    15: 'green,yellowRight',
    16: 'red,redRight',
    17: 'yellow,redRight',
    18: 'green,redRight',
    19: 'red,greenLeft',
    20: 'yellow,greenLeft',
    21: 'green,greenLeft'
}


pipeline_labels_to_new_labels = {
    "red": {
        "circle":  1,
        "left_arrow": 7,
        "right_arrow": 4,
        "up_arrow": 0,
        "down_arrow": 0,
        "down_left_arrow": 0,
        "down_right_arrow": 0,
        "cross": 0,
        "OVERLAP": 0,
    },
    "amber": {
        "circle": 2,
        "left_arrow": 8,
        "right_arrow": 5,
        "up_arrow": 0,
        "down_arrow": 0,
        "down_left_arrow": 0,
        "down_right_arrow": 0,
        "cross": 0,
        "OVERLAP": 0,
    },
    "green": {
        "circle": 3,
        "left_arrow": 9,
        "right_arrow": 6,
        "up_arrow": 0,
        "down_arrow": 0,
        "down_left_arrow": 0,
        "down_right_arrow": 0,
        "cross": 0,
        "OVERLAP": 0,
    },
    "white": {
        "circle": 0,
        "left_arrow": 0,
        "right_arrow": 0,
        "up_arrow": 0,
        "down_arrow": 0,
        "down_left_arrow": 0,
        "down_right_arrow": 0,
        "cross": 0,
        "OVERLAP": 0,
    },
    "unknown": {
        "unknown": 0,
        "OVERLAP": 0,
    }
}

separate_tls_to_combined_tls = {
    1: {
        6: 10,
        5: 13,
        4: 16,
        9: 19,
    },
    2: {
        6: 11,
        5: 14,
        4: 17,
        9: 20,
    },
    3: {
        6: 12,
        5: 15,
        4: 18,
        9: 21,
    }
}
