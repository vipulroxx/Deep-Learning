g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1], g.readlines()))
print(reviews[0])
g.close()
