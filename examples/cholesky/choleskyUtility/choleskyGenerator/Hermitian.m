function M = Hermitian(mSize)
V = single(10*rand(2*mSize) - 5);
M = single(zeros(mSize,mSize));
j = 1;
for k =1:mSize
    for h=k:mSize
        M(k,h) = single(V(j) +i*V(j+1));
        j=j+2;
    end
end

for k = 2:mSize
    for h=1:k
        M(k,h) = single(M(h,k)');
    end
end

end
