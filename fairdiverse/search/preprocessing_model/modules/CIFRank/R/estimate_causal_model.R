# code for estimating the causal model
# @article{yang2020causal,
#   title={Causal intersectionality for fair ranking},
#   author={Yang, Ke and Loftus, Joshua R and Stoyanovich, Julia},
#   journal={arXiv preprint arXiv:2006.08688},
#   year={2020}
# }
# License
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#if(!require('mma'))
install.packages("mma", repos = "https://cloud.r-project.org")
library(mma)

# CMake was not found on the PATH. Please install CMake:
# ERROR: configuration failed for package ‘nloptr’
save_mediation_results <- function (res, group_list, out_path){
    df_total <- res$bin.result$results$total.effect
    df_direct <- res$bin.result$results$direct.effect
    df_indirect <- res$bin.result$results$indirect.effect

    cols = list()
    for(i in 1:length(names(df_indirect)))
    {
        cols[i] <- strsplit(names(df_indirect)[i], 'pred')[[1]][2]
    }
    for(i in 1:length(group_list))
    {
        if(group_list[i] %in% cols)
        {
            col <- paste0("pred", group_list[i], sep='')
        }
        else{
            col <- "pred"
        }

        file_name <- paste0(group_list[i], "_med.csv")
        Metric <- c("Indirect Effect", "Direct Effect", "Total Effect")
        Estimate <- c(df_indirect[[col]][, "y1.all"]['est'], df_direct[, paste("y1.", col, sep='')]['est'], df_total[, paste("y1.", col, sep='')]['est'])
        df <- data.frame(Metric, Estimate)
        write.csv(df, file=paste(out_path, file_name, sep='/'))
    }
}

get_mediators <-function (data_i, MED){
    return(unlist(MED))
}


estimate_causal_model <- function (data_i, IV, DV, MED, control, out_path){
    # Create Causal Model
    data_i[, IV] <- as.factor(data_i[, IV])

    group_list<- unique(data_i[[IV]])
    group_list<-group_list[group_list != control]
    #print("Mediation...")
    # https://www.rdocumentation.org/packages/mma/versions/10.6-1/topics/mma
    med_cols = get_mediators(data_i, MED)


    check_med <- data.org(x=data_i[med_cols],y=data_i[, DV],pred=data_i[, IV], mediator=med_cols, predref=control)
    if(is.character(check_med)){
       mediators <- NULL
    }
    else{
         mediators = names(check_med$bin.results$contm)
    }

    if(is.null(mediators)) {
        mediators = c('NULL')
        df <- data.frame(mediators)
        file_name = "identified_mediators.csv"
        write.csv(df, file=paste(out_path, file_name, sep='/'))

        model.str = paste0(DV, "~", IV, '-1')
        form1 = as.formula(model.str)
        model.dv_iv <- lm(form1, data = data_i)


        write.csv(data.frame(summary(model.dv_iv)$coefficients), file=paste(out_path, paste0(model.str, '.csv', sep=''), sep='/'))
    }
    else{

        med_i<-med(data=check_med)

        df <- data.frame(mediators)
        file_name = "identified_mediators.csv"
        write.csv(df, file=paste(out_path, file_name, sep='/'))

        for(i in 1:length(mediators))
        {
            model.str = paste0(mediators[[i]], "~", IV, '-1')
            form2 = as.formula(model.str)
            model.iv_med <- lm(form2, data=data_i)
            write.csv(data.frame(summary(model.iv_med)$coefficients), file=paste(out_path, paste0(model.str, '.csv', sep=''), sep='/'))
        }
        }
}