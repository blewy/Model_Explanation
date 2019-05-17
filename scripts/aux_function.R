
require(kableExtra)
require(gridExtra)
require(grid)

#calculate ROC (https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
calculate_roc <- function(verset, cost_of_fp, cost_of_fn, n=100) {
  
  tp <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$obs == 1)
  }
  
  fp <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$obs == 0)
  }
  
  tn <- function(verset, threshold) {
    sum(verset$predicted < threshold & verset$obs == 0)
  }
  
  fn <- function(verset, threshold) {
    sum(verset$predicted < threshold & verset$obs == 1)
  }
  
  tpr <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$obs == 1) / sum(verset$obs == 1)
  }
  
  fpr <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$obs == 0) / sum(verset$obs == 0)
  }
  
  cost <- function(verset, threshold, cost_of_fp, cost_of_fn) {
    sum(verset$predicted >= threshold & verset$obs == 0) * cost_of_fp + 
      sum(verset$predicted < threshold & verset$obs == 1) * cost_of_fn
  }
  fpr <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$obs == 0) / sum(verset$obs == 0)
  }
  
  threshold_round <- function(value, threshold)
  {
    return (as.integer(!(value < threshold)))
  }
  #calculate AUC (https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)
  auc_ <- function(verset, threshold) {
    auc(verset$obs, threshold_round(verset$predicted,threshold))
  }
  
  roc <- data.frame(threshold = seq(0,1,length.out=n), tpr=NA, fpr=NA)
  roc$tp <- sapply(roc$threshold, function(th) tp(verset, th))
  roc$fp <- sapply(roc$threshold, function(th) fp(verset, th))
  roc$tn <- sapply(roc$threshold, function(th) tn(verset, th))
  roc$fn <- sapply(roc$threshold, function(th) fn(verset, th))
  roc$tpr <- sapply(roc$threshold, function(th) tpr(verset, th))
  roc$fpr <- sapply(roc$threshold, function(th) fpr(verset, th))
  roc$cost <- sapply(roc$threshold, function(th) cost(verset, th, cost_of_fp, cost_of_fn))
  roc$auc <-  sapply(roc$threshold, function(th) auc_(verset, th))
  
  return(roc)
  
}


plot_roc <- function(roc, threshold, cost_of_fp, cost_of_fn) {
  library(gridExtra)
  
  norm_vec <- function(v) (v - min(v))/diff(range(v))
  
  idx_threshold = which.min(abs(roc$threshold-threshold))
  
  col_ramp <- colorRampPalette(c("green","orange","red","black"))(100)
  col_by_cost <- col_ramp[ceiling(norm_vec(roc$cost)*99)+1]
  p_roc <- ggplot(roc, aes(fpr,tpr)) + 
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=2, alpha=0.5) +
    labs(title = sprintf("ROC")) + xlab("FPR") + ylab("TPR") +
    geom_hline(yintercept=roc[idx_threshold,"tpr"], alpha=0.5, linetype="dashed") +
    geom_vline(xintercept=roc[idx_threshold,"fpr"], alpha=0.5, linetype="dashed")
  
  p_auc <- ggplot(roc, aes(threshold, auc)) +
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=2, alpha=0.5) +
    labs(title = sprintf("AUC")) +
    geom_vline(xintercept=threshold, alpha=0.5, linetype="dashed")
  
  p_cost <- ggplot(roc, aes(threshold, cost)) +
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=2, alpha=0.5) +
    labs(title = sprintf("cost function")) +
    geom_vline(xintercept=threshold, alpha=0.5, linetype="dashed")
  
  sub_title <- sprintf("threshold at %.2f - cost of FP = %d, cost of FN = %d", threshold, cost_of_fp, cost_of_fn)
  # 
  grid.arrange(p_roc, p_auc, p_cost, ncol=2,sub=textGrob(sub_title, gp=gpar(cex=1), just="bottom"))
  
}



plot_confusion_matrix <- function(verset,threshold=0.5, sSubtitle) {
  tst <- data.frame(Predicted=if_else(verset$predicted<=threshold,0,1), True=verset$obs)
  opts <-  c("Predicted", "True")
  #names(tst) <- opts
  cf <- plyr::count(tst)
  cf[opts][cf[opts]==0] = "NO"
  cf[opts][cf[opts]==1] = "Yes"
  
  ggplot(data =  cf, mapping = aes(x = True, y = Predicted)) +
    labs(title = "Confusion matrix", subtitle = sSubtitle) +
    geom_tile(aes(fill = freq), colour = "grey") +
    geom_text(aes(label = sprintf("%1.0f", freq)), vjust = 1) +
    scale_fill_gradient(low = "lightblue", high = "blue") +
    theme_bw() + theme(legend.position = "none")
}


mplot_density <- function(tag, 
                          score, 
                          thresh = 5,
                          model_name = NA, 
                          subtitle = NA, 
                          save = FALSE, 
                          subdir = NA,
                          file_name = "viz_distribution.png") {
  
  # require(ggplot2)
  # require(gridExtra)
  # require(scales)
  
  if (length(tag) != length(score)) {
    message("The tag and score vectors should be the same length.")
    stop(message(paste("Currently, tag has",length(tag),"rows and score has",length(score))))
  }
  
  if (length(unique(tag)) <= thresh) {
    
    out <- data.frame(tag = as.character(tag),
                      score = as.numeric(score))
    
    if (max(out$score) < 1) {
      out$score <- score * 100
    }
    
    p1 <- ggplot(out) + theme_minimal() +
      geom_density(aes(x = score, group = tag, fill = as.character(tag)), 
                   alpha = 0.6, adjust = 0.25) + 
      guides(fill = guide_legend(title="Tag")) + 
      xlim(0, 100) + 
      labs(title = "Classification Model Results",
           y = "Density by tag", x = "Score")
    
    p2 <- ggplot(out) + theme_minimal() + 
      geom_density(aes(x = score), 
                   alpha = 0.9, adjust = 0.25, fill = "deepskyblue") + 
      labs(x = "", y = "Density")
    
    p3 <- ggplot(out) + theme_minimal() + 
      geom_line(aes(x = score, y = (1 - ..y..), color = as.character(tag)), 
                stat = 'ecdf', size = 1) +
      geom_line(aes(x = score, y = (1 - ..y..)), 
                stat = 'ecdf', size = 0.5, colour = "black", linetype="dotted") +
      ylab('Cumulative') + xlab('') + guides(color=FALSE)
    
    if(!is.na(subtitle)) {
      p1 <- p1 + labs(subtitle = subtitle)
    }
    
    if(!is.na(model_name)) {
      p1 <- p1 + labs(caption = model_name)
    }
    
    if (!is.na(subdir)) {
      dir.create(file.path(getwd(), subdir), recursive = T)
      file_name <- paste(subdir, file_name, sep="/")
    }
    
    if(save == TRUE) {
      png(file_name, height = 1800, width = 2100, res = 300)
      grid.arrange(
        p1, p2, p3, 
        ncol = 2, nrow = 2, heights = 2:1,
        layout_matrix = rbind(c(1,1), c(2,3)))
      dev.off()
    }
    
    return(
      grid.arrange(
        p1, p2, p3, 
        ncol = 2, nrow = 2, heights = 2:1,
        layout_matrix = rbind(c(1,1), c(2,3))))
    
  } else {
    
    df <- data.frame(
      rbind(cbind(values = tag, type = "Real"), 
            cbind(values = score, type = "Model")))
    df$values <- as.numeric(as.character(df$values))
    
    p <- ggplot(df) + theme_minimal() +
      geom_density(aes(x = values, fill = as.character(type)), 
                   alpha = 0.6, adjust = 0.25) + 
      labs(y = "Density", x = "Continuous values") +
      scale_x_continuous(labels = comma) +
      guides(fill = guide_legend(override.aes = list(size=1))) +
      theme(legend.title=element_blank(),
            legend.position = "top")
    
    if(!is.na(model_name)) {
      p <- p + labs(caption = model_name)
    }
    
    if(!is.na(subtitle)) {
      p <- p + labs(subtitle = subtitle)
    }  
    
    if (!is.na(subdir)) {
      dir.create(file.path(getwd(), subdir), recursive = T)
      file_name <- paste(subdir, file_name, sep="/")
    }
    
    if (save == TRUE) {
      p <- p + ggsave(file_name, width = 6, height = 6)
    }
    return(p)
  }
}

