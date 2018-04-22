def loadMovieList():
    """
    reads the fixed movie list in movie.txt
    and returns a cell array of the words in movieList.
    """

    #  Read the fixed movieulary list
    fid = open('movie_ids.txt', encoding='ISO-8859-1')

    # Store all movies in cell array movie{}
    n = 1682  # Total number of movies

    movieList = [None] * n
    for i in range(n):
        # Read line
        line = fid.readline()
        # Word Index (can ignore since it will be = i)
        movieName = str.split(line)[1:]
        # Actual Word
        movieList[i] = ' '.join(movieName)

    fid.close()

    return movieList
