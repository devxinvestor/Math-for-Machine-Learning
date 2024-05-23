using Pkg
using SparseArrays
include("wikipedia_corpus.jl")
include("kmeans.jl")
using Main.Kmeans
using PyPlot
pygui(true)
global(legend=true)

"---------------------Two Clusters---------------------"
J1 = []
J2 = []
k = 2

centroids, labels, losses = kmeans(article_histograms, k)
push!(J1, losses...)

centroids, labels, losses = kmeans(article_histograms, k)
push!(J2, losses...)

plot(1:length(J1), J1, color = "green", linewidth=2.0), plot(1:length(J2), J2, color = "blue", linewidth=2.0),
xlabel("Iteration"), ylabel("Loss (J)"), title("Euclidean distance from the ith data point to each centroid (k = $k)")

clf()
# With the two clusters in particular, the J values tend to be higher overall. Also, the difference
# between each of the J values is not as significant as some of the other clusters.
# When you get to the second k-means run, the k values also seem to be smaller, meaning that the
# k-means is more optimized.

"---------------------Five Clusters---------------------"
J1 = []
J2 = []
k = 5

centroids, labels, losses = kmeans(article_histograms, k)
push!(J1, losses...)

centroids, labels, losses = kmeans(article_histograms, k)
push!(J2, losses...)

plot(1:length(J1), J1, color = "green", linewidth=2.0), plot(1:length(J2), J2, color = "blue", linewidth=2.0),
xlabel("Iteration"), ylabel("Loss (J)"), title("Euclidean distance from the ith data point to each centroid (k = $k)")

clf()

# With the five clusters, the J values drop off significantly after every iteration.
# The distance between the first run through of k-means and the second one are less spaced for
# 5 clusters than 2 clusters. Additionally, when you get to the second k-means run, the k values seem to 
# be smaller, meaning that the k-means is more optimized.

"---------------------Ten Clusters---------------------"
J1 = []
J2 = []
k = 10

centroids, labels, losses = kmeans(article_histograms, k)
push!(J1, losses...)

centroids, labels, losses = kmeans(article_histograms, k)
push!(J2, losses...)

plot(1:length(J1), J1, color = "green", linewidth=2.0), plot(1:length(J2), J2, color = "blue", linewidth=2.0),
xlabel("Iteration"), ylabel("Loss (J)"), title("Euclidean distance from the ith data point to each centroid (k = $k)")

clf()

# With the ten clusters, the J values have a steep drop off after the first iteration.
# The distance between the first run through of k-means and the second one are hardly differentiable,
# similar to the 5 clusters. Additionally, when you get to the second k-means run, the k values seem to 
# be smaller, meaning that the k-means is more optimized.

"---------------------K Value Investigation---------------------"

k = 5
centroids, labels, losses = kmeans(article_histograms, k)

centroids
labels
losses


article_titles[labels .== 1]
# Article titles for cluster 1
# The first 3 titles are "A Bar at the Folies-Bergère", "Alfred Sisley", and "Armand Guillaumin"
# The articles seem to have to do with artwork and paintings

article_titles[labels .== 2]
# Article titles for cluster 2
# The first 3 titles are "Amplitude modulation", "Amplitude-shift keying", and "Analog signal"
# The articles seem to have to do with radio stations and broadcasting

article_titles[labels .== 3]
# Article titles for cluster 3
# The first 3 titles are "Acid rain", "Albedo", and "Anemometer"
# The articles seem to have to do with the weather

article_titles[labels .== 4]
# Article titles for cluster 4
# The first 3 titles are "Convention on the Rights of Persons with Disabilities", "Food and Agriculture Organization", and "Headquarters of the United Nations"
# The articles seem to have to do with activist organizations

article_titles[labels .== 5]
# Article titles for cluster 5
# The first 3 titles are "Brock (Pokémon)", "Bulbasaur", "Deoxys"
# The articles seem to have to do with video games, particularly anime



dictionary[sortperm(centroids[1],rev=true)]
# Most common words of cluster 1 are painting, art, and paintings

dictionary[sortperm(centroids[2],rev=true)]
# Most common words of cluster 2 are signal, radio, and frequency

dictionary[sortperm(centroids[3],rev=true)]
# Most common words of cluster 3 are weather, wind, and pressure

dictionary[sortperm(centroids[4],rev=true)]
# Most common words of cluster 4 are nations, international, and member

dictionary[sortperm(centroids[5],rev=true)]
# Most common words of cluster 5 are pokemon, game, and games