clc
clear all

NumSol = 20
%% Find The Values of f and Select Solutions for Tournament

%solutions = 10 * rand(NumSol, 3)
solutions  = zeros(NumSol, 3);
fValues    = zeros(NumSol, 1);
Sol_Vals   = zeros(NumSol, 4);
tournament = zeros(NumSol, 2);
for i = 1:NumSol
  %x1 = [x1, solutions(i, 1)];
  %x2 = [x2, solutions(i, 1)];
  %x3 = [x3, solutions(i, 1)];
  x1 = 10 * rand(1);
  x2 = 10 * rand(1);
  x3 = 10 * rand(1);
  solutions(i, :) = [x1, x2, x3];
  
  
  %f = (x1(i)^2) + 2 * (x2(i)^2) + 3 * (x3(i)^2) + x1(i)*x2(i) + x2(i)*x3(i) -8*x1(i) -16*x2(i) - (32*x3(i)) +110;
  f = (x1^2) + 2 * (x2^2) + 3 * (x3^2) + x1*x2 + x2*x3 -8*x1 -16*x2 - (32*x3) +110;
  fValues(i) = f;
  Sol_Vals(i, :) = [x1, x2, x3, fValues(i)];
  
  sol1 = round((20 - 1)*rand(1) + 1)
  sol2 = round((20 - 1)*rand(1) + 1)
    
  while sol1 == sol2    
    sol1 = round((20 - 1)*rand(1) + 1)
    sol2 = round((20 - 1)*rand(1) + 1)
  end
        
    tournament(i, :) = [sol1, sol2];
  
  
end


Sorted = sortrows(Sol_Vals, 4)


%% Tournament

New_Population = zeros(NumSol, 1);

for i = 1:NumSol
    
    if rand(1) > 0.1
    New_Population(i) = min([fValues(tournament(i, 1)), fValues(tournament(i, 2))])
    
    else
    New_Population(i) = max([fValues(tournament(i, 1)), fValues(tournament(i, 2))])

    end
    
end

%% Calculate the Occurances of Previous Best Five Solutions in the New Population
count = zeros(5, 1)

for i = 1:5
   
    count(i) = sum(New_Population(:) == Sorted(i, 4))
    
end




fprintf('Top Five Solutions in Current Population: \n') 
disp(Sorted(1:5, 1:3))
fprintf('Corresponding f Values: \n') 
disp(Sorted(1:5, 4))
fprintf('Number of Occurances of Each Solution in The New Population: \n')
disp(count)



