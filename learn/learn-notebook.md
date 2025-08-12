how left join work
if a and b join together, a will keep all the rows, but for matching rows in b will be added to a

the data has 320k data without underwriting report

the data has 18k data that do not match the loan data, as these application not make to the loan table

what can we see from the histogram
left and right skewed distribution can be detected
apr or annual percentage rate is left skewed with most loan having higher rate
loan amount is right skewed showing the loan amount is not alot from the customer most between 1000 dollars
the original scheduled payment amount is also right skewed as most loan amount is low, this is expected
the number of paid off are close to zero so most of the loan has not being paid off
the lead cost is skewed to the right, mean customer cost less to acquire
the clear fraud score distribution is slightly skewed to the left, suggesting the credibility or the borrower

what we can see from the bar chart
most of the loan is not from collection (custom plan if they face problem repaying back)
the payment mostly are weekly and biweekly
the customer lead are mostly from bymandatory as well (there are ping tree purchased which need bank verification)
we can also see most payment are either cancelled or checked, so there are attempted payment and also failed payment
loan status will be check discuss later due to its usage as target variable
for state we might need to change to categorical during the training as the dimension are quite high, we can also consider grouping by region if possible

how to make the target variable
external or internal collection is bad as they are default loan, efforts are made to recover the fund
returned item is also bad due to missing payment like lack of fund
charged off is also bad as the debt being written off as loss
settled bankruptcy also bad due to default
settlement paid off also bad while some money being recovered less than what being owed
paid off loan is good, fully repaid
new loan is good as we don't know very well yet, but we let it pass so it could be good for now
pending paid off is also good
withdrawn application is excluded as the loan is not being withdrawn
rejected is excluded as well as we want to find new ways to find risky loan, adding them will not help in identifying the riskier loans
voided status are excluded as the loan not being funded
pending status also excluded due to the incomplete process

the scatterplot show what
no clear separation for good and bad loan, the relationship can be more complex than we think it is

the bar chart tell what
loan amount for good and bad are almost similar with same median
apr for good and bad loan also identical, the good loan has bigger spread than the bad. for bad loan we can see the outlier some of the loan is quite low, this could be some specific decision that need to be investigate further
clear fraud score for good loan also higher which is good as the it indicates borrower credibility
for paid off in good and bad loan both are low, so not much history in paying the loan, this feature probably not helpful in decision making later on

what we can see from the heatmap
we will only see the top 5 highest and bottom 5
for top loan amount and the scheduled payment as you get more and pay more so that make sense
seven day underwriting and fifteen day underwriting, so many inquiry in the past 7 will also be aggragated to the past 15, so make sense 
similar for 15 and 30 days cases
similar for 30 and 90 days cases
principal and payment amount also correlated again if we get more we pay back more

