# TEST

setup='{
    "fl_algorithm": "fed_avg",
    "data_distribution": "rare_on_rare",
    "clients": {
        "count": 10
    },
    "algorithm_params": {
        "global_rounds_num": 7,
        "local_epochs": 3
    },
    "distribution_params": {
        "sized_distribution": "exp",
        "rare_clients":2,
        "rare_data_count": 4,
        "scale": 5,
        "random_state": 2473
    },
    "saver_directory": "/home/cwd/run_data"
}'

# ================================================
#                   calls
# ================================================

curl -X POST localhost:8000/forward_distribution \
     -H "Content-Type: application/json" \
     -d "$setup"

# curl -X POST localhost:8000/forward \
#      -H "Content-Type: application/json" \
#      -d "$setup"
