import os
import random
import re
import sys

sys.setrecursionlimit(11000)
DAMPING = 0.85
SAMPLES = 10000

'''
{'1': {'2'}, '2': {'3', '1'}, '3': {'2', '5', '4'}, '4': {'2', '1'}, '5': set()}
{'1': {'2'}, '2': {'3', '1'}, '3': {'2', '4'}, '4': {'2'}, '5': {'6'}, '6': {'5', '7'}, '7': {'6', '8'}, '8': {'6'}}
{'1': {'2'}, '2': {'3', '1'}, '3': {'2', '4'}, '4': {'2'}}
{'1': {'2', '3'}, '2': {'3', '4', '1'}, '3': {'5', '4'}, '4': {'6', '2', '1', '3'}, '5': {'3'}, '6': {'2', '1', '3'}}
'''

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(
        {'1': {'2'}, '2': {'3', '1'}, '3': {'2', '4'}, '4': {'2'}},
        DAMPING,
        SAMPLES
    )

    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank({'1': {'2'}, '2': {'1', '3'}, '3': {'4', '2'}, '4': {'2'}}, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    N = len(corpus)
    d = damping_factor
    first_term = (1 - d) / N
    nLinks_of_page = len(corpus.get(page))
    return_dict = { page: first_term }

    for link in corpus.get(page):
        dest_probability = (d / nLinks_of_page)
        return_dict[link] = first_term + dest_probability

    for link in corpus:
        if not link in corpus.get(page):
            return_dict[link] = first_term
    
    return return_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    nInit = n
    pages_array = []
    r = random.randrange(0, len(corpus))
    for page in corpus:
        pages_array.append(page)
    random_page = pages_array[r]
    page_sample_array = []
    page_sample_array = rec(corpus, random_page, damping_factor, n, page_sample_array, nInit)

    retval = {}
    for page in corpus:
        retval[page] = page_sample_array.count(page) / nInit

    return retval


def rec(corpus, random_page, damping_factor, n, page_sample_array, nInit):

    dist = transition_model(corpus, random_page, damping_factor)
    n -= 1
    if n == 0:
        return page_sample_array

    dist_array = []
    for p in dist:
        dist_array.append([p, dist[p]])

    pages = []
    weights = []
    for i, item in enumerate(dist_array):
        pages.append(item[0])
        weights.append(item[1])
    
    sample = random.choices(pages, weights = weights, k = 1)
    page_sample_array.append(sample[0])
    return rec(corpus, sample[0], damping_factor, n, page_sample_array, nInit)


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Set up initial transition model, all equaly weighted
    N = 0 #total number of pages
    for page in corpus:
        N += 1
    retval = {}
    for page in corpus:
        retval[page] = 1 / N
    # Pick a random page
    pages_array = []
    r = random.randrange(0, len(corpus))
    for page in corpus:
        pages_array.append(page)
    random_page = pages_array[r]

    # Intialize page sample array
    page_sample_array = []
    page_sample_array = rec(corpus, random_page, damping_factor, 2, page_sample_array, 1)

    # Initialize PREVIOUS_RETVAL
    PREVIOUS_RETVAL = {}
    for page in corpus:
        PREVIOUS_RETVAL[page] = 0

    # Initialize False = convergence 
    convergence = False
    n = 0
    TEMPORARY = 100
    while not convergence:
        # Pick a random page
        pages = []
        weights = []
        for item in retval:
            pages.append(item)
            weights.append(retval[item])
        
        sample = random.choices(pages, weights = weights, k = 1)

        # get the next retval
        n += 1
        page_sample_array = rec(corpus, sample[0], damping_factor, 2, page_sample_array, 1)
        retval = {}
        for page in corpus:
            retval[page] = page_sample_array.count(page) / ( n + 1 )

        # COMPARE WITH PREVIOUS_RETVAL
        is_all_less_than_point001 = True
        for page in retval:
            if abs(PREVIOUS_RETVAL[page] - retval[page]) >= .001:
                is_all_less_than_point001 = False

        if not is_all_less_than_point001:
            PREVIOUS_RETVAL = retval.copy()

        if is_all_less_than_point001:
            sum = 0
            for p in retval:
                sum += retval[p]
            # Compare each page of 
            convergence = True
            return retval

    return None

if __name__ == "__main__":
    main()
