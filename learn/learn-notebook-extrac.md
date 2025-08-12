- All the payments are being matched
- They are also null values in the dataset which need to be filled

- There are about 320k data without underwriting report
- There are also about 18k underwriting report that do not correspond to the loan data, they are applications that never made to the `loan` table

- Loan amount seems rightly skewed
- Transformation is needed for the variables as well
- Most of the `nPaidOff` are 0 which means majority of borrowers in the dataset have never previously paid off a loan from the lending institution. 
- Most of the customers have low lead cost meaning they do not cost much to acquire

- Higher number of loan is `isCollection` = False, so the customer is not from custom plan where they face problem repaying
- Most payment frequency are either weekly or bi-weekly
- Most customer has lead type `byMandatory` which is a lead that has been purchased from a ping tree and requires a bank verification to be completed before a loan can be approved
- Most of the payment status are either cancelled or checked suggested there are successful payments however there are also a significant amount of scheduled payment being called off and not attempted
- We might need some handling on the category data to avoid high dimensionality like `loanStatus` and `state`

**We will make a `isRisky` column below to denote loan risk.**
- For bad loan here are some of the examples:
    - External Collection, Internal Collection, Returned Item, Charged Off, Charged Off Paid Off, Settled Bankruptcy, Settlement Paid Off, Settlement Pending Paid Off

- For good loan here are some of the examples:
    - Paid Off Loan, Pending Paid Off

- Excluded cases
    - Withdrawn Application, Rejected, Voided, Pending Rescind, Pending Application Fee and so on
    - Why?
        - To avoid confusing the model and causes data leakage

**Explanations**
- External Collection / Internal Collection: The loan is in default and efforts are being made to recover the funds. This is the most definitive "bad" status.
- Returned Item: The customer missed a payment due to insufficient funds. This is a very strong early indicator of high risk.
- Charged Off / Charged Off Paid Off: The company has written the debt off as a loss. This is the worst financial outcome.
- Settled Bankruptcy: The customer declared bankruptcy, a clear default event.
- Settlement Paid Off / Settlement Pending Paid Off: The company agreed to accept less than the full amount owed. While some money was recovered, it was still a loss compared to the original agreement. This is a "bad" outcome.
- Paid Off Loan: The ideal outcome. The loan was fully repaid.
- New Loan: The loan is currently active and, presumably, not yet showing signs of distress. For a predictive model, these are typically considered "good" at the time of data collection.
- Pending Paid Off: The loan is about to be successfully completed.
- Withdrawn Application: The customer never took out the loan. There is no repayment behavior to learn from. Exclude these.
- Rejected: The loan was rejected by MoneyLion's existing underwriting rules. If you include these as "bad," your model will simply learn to copy the old rules, not to find new patterns of risk in approved loans. This is a form of target leakage. You want to build a model that assesses risk on applications that would be or were approved. Exclude these.
- All Voided Statuses (Credit Return Void, Customer Voided New Loan, etc.): The loan was approved but never funded or was canceled immediately. Like withdrawn applications, there is no repayment behavior. Exclude these.
- All other Pending Statuses (Pending Rescind, Pending Application Fee, etc.): The loan process is not complete. There is no final outcome to learn from. Exclude these.

- Almost half the data is excluded. They will not be included in the model training to avoid confusing the model and causing data leakage.
- Moreover, we can see that the class is not imbalanced and we have an almost balanced dataset.

```txt
Total Samples: 351,150 + 320,737 = 671,887
Percentage of Class 1.0: (351,150 / 671,887) * 100 ≈ 52.3%
Percentage of Class 0.0: (320,737 / 671,887) * 100 ≈ 47.7%
```

**Scatterplot**
- No clear separation is detected in the plot
- Loan risk relationship with these variables are likely to be more complex

**Barchart**
- Median loan amount is identical for good and bad loans with big overalap
- Median APR is also identical for good and bad loans with a bigger distribution for good loans and more outliers for bad loans with lower APR
    - A significant number of them were issued with surprisingly low rates. This could be due to specific underwriting decisions, collateral-based loans, or other factors that don't fit the typical high-risk profile.
- Median fraudscore for good loan is higher and the distribution of it also shifted higher than bad loans
- Most of the applications have no history of paid-off loans, this variable probably does not contribute much to the model

- We will extract the highest 5 and lowest 5 correlation variables for interpretation
- It appears no variable has a particular strong correlation with `isRisky` suggesting the complexity of identifying risky loans

**The Top**
- `loanAmount and originallyScheduledPaymentAmount`
    - Larger amount would result in larger repayment amount
- `.underwritingdataclarity.clearfraud.clearfraudinquiry.sevendaysago and .underwritingdataclarity.clearfraud.clearfraudinquiry.fifteendaysago`
    - If consumer make many inquiries in the past 7 days it will also reflected in the 15 day count
- `.underwritingdataclarity.clearfraud.clearfraudinquiry.thirtydaysago and .underwritingdataclarity.clearfraud.clearfraudinquiry.fifteendaysago`
    - Logical as 15 day inquiries are a subset of 30 day inquiries
- `principal and paymentAmount`
    - Higher principal portion of a payment will lead to higher total payment amount
- `.underwritingdataclarity.clearfraud.clearfraudinquiry.thirtydaysago and .underwritingdataclarity.clearfraud.clearfraudinquiry.ninetydaysago`
    - Inquiries made in the past 30 days is part of the total inquiries made in the last 90 days

**The Bottom**
- `.underwritingdataclarity.clearfraud.clearfraudinquiry.sevendaysago and clearfraudscore`
    - More recent inquiries may be seen as a sign of higher risk reducing the fraud score
- `.underwritingdataclarity.clearfraud.clearfraudinquiry.fifteendaysago and clearfraudscore`
    - Similar to the 7 days ago, but with a slightly lower correlation
    - Recent credit seeking behavior is a factor in determining default risk
- `.underwritingdataclarity.clearfraud.clearfraudinquiry.ninetydaysago and clearfraudscore`
    - Higher number of inquiries in the last 90 days associated with lower fraud score
- `.underwritingdataclarity.clearfraud.clearfraudinquiry.thirtydaysago and clearfraudscore`
    - Similar to the 90 days ago, but with a slightly lower correlation
- `apr and loanAmount`
    - Larger loan are considered less risky
    - They can be given as lower interest to attract people borrowing them

**The Missingness**
- For a tree-based model like LightGBM, it's often more powerful to make the "missingness" itself a piece of information rather than trying to perfectly guess the original value. The model can learn if the absence of a value is correlated with risk.
- We will use median for numerical variables as financial data is often skewed and median is a safer and more representative choice
- For numerical column with missing values we will also added a `is_missing` indicator to flag the missingness
- Furthermore, for categorical variables we will just impute them with a special class like `unknown` or `missing` to indicate the missingness

### Feature Engineering

**What to do:**
- Engineer the date columns like finding date differences, extracting year, month, day, etc.

- `days_diff`
    - Very Short Duration: A very quick origination might indicate a straightforward, low-risk applicant, or potentially a less thorough underwriting process.
    - Very Long Duration: A long delay could signify that the underwriting process was complex, requiring more documentation or verification, which could be associated with higher risk. It might also indicate that the applicant was initially hesitant.
- Data transformation
    - Applicants applying at the end of the month might be facing financial shortfalls. Similarly, applications on weekends versus weekdays could exhibit different risk profiles.

### Imputations

**What to do:**
- For numerical variables, we will use median to fill the missing values and add a `is_missing` indicator column to flag the missingness.
- For categorical variables, we will impute them with a special class like `unknown` or `missing` to indicate the missingness.
- We will verify no missing values remain in the dataset before saving the processed data.

## Modeling

**What to do:**
- Train test and validation splitting of 70 | 15 | 15
- Build a LightGBM model to predict risk of loan applications
- Evaluate the model performance using appropriate metrics
- Save the model

**Note on Splitting Strategy**
- Due to the data being historical ranging from 2014 to 2017, I propose a time-based split to simulate a real-world scenario as:
    - Economic shift might happen over time
    - Applicant behavior may change over time
    - MoneyLion marketing might attract different types of applicants over time
- If we use a split without considering time we might leak information from the future into the past

**What Happened:**
- We use each trial to seek for the best parameters
- We use stratified k-fold cross validation so the class distribution is preserved in each fold 
- We use the best parameters to train the model on the training set
    - 5 folds we will have 1 set holdout the other for training, and then repeat for 5 times
- Although we have a large round to give the model to learn, but we will have early stopping of 50 if the performance does not improve for 50 rounds
- The optuna will monitor the AUC and optimize the parameter accordingly

**What Happened:**
- We will copy the best parameter from optuna here
- We will then train the model again

**What Happened:**
- For our hidden test set performance the accuracy is low and meaning it gets prediction wrong quite often
- Nonetheless, the model is good at finding risky loans as seen from the recall. With a score of `0.82` it identifies the risky loans 82% of the time which is good.
- The risky loan precision of `0.45` for risky loan means that for all the predicted risky loans only 45% are actually true, that means a great amount of good loan are falsely flagged
- With good loans recall is `0.42` suggesting the actually good one the model can only get 42% of time which can cause a good amount of customers to be wrongly flagged.
- This can lead to lost revenue

**What to do:**
- We use a 0.5 threshold to classify the loans as risky or not risky so we might need to adjust it 
- We will also see the precision-recall curve to see how we need to make the adjustment
- Furthermore, is false negative more costly or false positive more costly
    - False negative (falsely identify as not risky) will lead to financial loss
        - This is more costly as risky transactions are allowed to proceed
        - In our case we might be lending out and cannot recover the money
    - False positive (falsely identify as risky) will lead to operational cost and lost opportunity
        - Less costly where the team will investigate the transaction and customer might change services as their service is being blocked

**What Happened:**
- ROC Curve
    - We can see the model has a good discriminative ability better than random guessing
    - However, there's a gap between the hidden test set and validation set where model might slightly overfit train and validation data
    - The test set only contains post-2017 data and the model does not generalize well in that case is a realistic outcome (economic shift, people behavior change, etc.)
- Precision-Recall Curves
    - From the curve, if we want a recall of 0.8 (identified 80% of risky loans) we will have a precision of close of 0.5, this corresponds to the report we created earlier 
- Threshold
    - We plot it using validation set as the test set should be hidden and unknown all the time
    - We can see a significant overlapping region of confusion in between the two class
    - For a large orange (risky) area to the right, they are risky so we have high recall meaning we correctly capture them
    - But for a large blue (good) are to the right we also flagged them as risky (false positive) which leads to a poor precision for the model

**What Happened:**
- For strategy A, we manage to identify cloe to 76.5% of the risky loans 
- For strategy B, we follow the constraint of catching at least 70% risky loans. We can see in this strategy the precision increase to 67% from 63.3 %.
- We will do some simulation to see which strategy is better. But Strategy B seems better, although we miss 5% of the risky loans we get more precise in the flagging process, that means less human review and less customer being turned away. But in business, high recall is often prioritized due to their costliness

**What Happened:**
- The first plot shows a tug of war between recall and precision as threshold increased
    - The recall will decrease and less risky loans are identified, but the flagged risky loans will be more precise
    - The f1 score is just a balance between the two and strategy A seems to identify that
- The second plot is just showing that strategy A has higher recall but lower precision while strategy B has lower recall but higher precision 
- the third plot shows how the loan will be affected by the threshold, if we use strategy B
    - we will have more false negative (risky flagged as good) but the false positive reduced which save operation time and provide better customer experience

**What Happened:**
- Note the result might be different if the loss rate and review cost is different.
- Using strategy in this scenario we earn extra 14.7 million thant strategy B
- When we choose strategy B we did not quantify financial impact where we can see it is very costly to miss a risky loan so we should get aim for higher recall

```
cost of false negative = 4000
ost of false positive = 50 (4000 / 50 = 80 times less)
```

- Although we spend more on review in strategy A, but the ROI is higher as well as the costliness of defaulting loan we should aim for higher recall

**What Happened:**
- `paymentStatus` is the most important
    - It is the result of ACH attempt (automatic clearing house, electronic transfer from bank to credit unions)
    - Its outcome indicate the ability or willingness of the customer to pay
- `fpStatus`
    - This is the result of the first payment
    - An early success will shows a customer reliability and potential trajectory
- `nPaidOff`
    - How many loans this customer has paid off before
    - Good history will help with the customer creditworthiness
- `clearfraudscore`
    - Suspicious profile will have higher risks
- `state`
    - Location and demographics can influence risk as well
- `paymentReturnCode`
    - ACH error code why payment failed
    - We learn more about the reasons
- `leadType`
    - Different lead types will have different risk profiles
    - Return customer can be less risky then a new leads from `ping tree`

**What is a ping tree?**
- Sarah wants to loan so she applies online (lead generator) which creates a ping
- The tree which is the lender will get her information and the lender that is most willing to take the risk will accept her application
