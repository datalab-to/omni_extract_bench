from omni_extract_bench.score import grade, explain
import datasets 

def main():
    dataset = datasets.load("datalab-to/omni_extract_bench")
    
    for element in dataset:
        breakpoint()
    
if __name__ == "__main__":
    main()