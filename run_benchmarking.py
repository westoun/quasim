#!/usr/bin/env python3

from benchmark import run_time_benchmark, run_result_benchmark


if __name__ == "__main__":
    run_time_benchmark(qubit_num=3)
    run_time_benchmark(qubit_num=4)
    run_time_benchmark(qubit_num=5)
    run_time_benchmark(qubit_num=6)
    run_time_benchmark(qubit_num=7)
    run_time_benchmark(qubit_num=8)
    run_result_benchmark(qubit_num=3)
    run_result_benchmark(qubit_num=5)
    run_result_benchmark(qubit_num=7)
