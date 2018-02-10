function out = feedForward(in, Ws, activation)

nLayers = length(Ws);

ai = in;

for i = 2:nLayers
  ai = activation(Ws{i-1}*ai);
  if (i ~= nLayers) % bias nodes not connected to previous layer
    ai(1) = 1;
  end
end

out = ai;

end