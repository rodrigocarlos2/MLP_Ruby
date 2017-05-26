
require 'rubygems'
require 'mlp'

a = MLP.new(:hidden_layers => [5], :output_nodes => 1, :inputs => 1)

6001.times do |i|
  a.train([0,0], [1])
  a.train([0,1], [0])
  a.train([1,0], [0])
  error = a.train([1,1], [1])
  puts "Error after iteration #{i}:\t#{error}" if i%200 == 0
end

puts "Test data"
puts "[0,0] = > #{a.feed_forward([0,0]).inspect}"
puts "[0,1] = > #{a.feed_forward([0,1]).inspect}"
puts "[1,0] = > #{a.feed_forward([1,0]).inspect}"
puts "[1,1] = > #{a.feed_forward([1,1]).inspect}"