# Запуск оценки RPC из репозитория Zhang & Xia (github.com/zmk-98/RRPM-IFEs) на панели PWT.
# Их код не копируется: файл RPC.R читается из склонированного репозитория, и в него на лету
# добавляются счётчик итераций и запись RSS по итерациям.
# Использование: Rscript zx_rpc_pwt.R <папка с репозиторием> <max_iter> <выходной префикс>
args <- commandArgs(TRUE); zx_dir <- args[1]; max_iter <- as.integer(args[2]); out <- args[3]
src <- readLines(file.path(zx_dir, "codes", "real data analysis", "RPC.R"))
i1 <- grep("^\\s*lambda = lambda\\s*$", src)[1]
src <- append(src, "  rss_hist <- c(); n_iter <- 0", after = i1)
i2 <- grep("if\\(abs\\(RSS-sum\\(resi\\^2\\)\\) < 1.0e-6\\)", src)[1]
src <- append(src, "      rss_hist <- c(rss_hist, sum(resi^2)); n_iter <- iter1", after = i2 - 1)
src <- sub("return(list(resi=resi,", "return(list(rss_hist=rss_hist, n_iter=n_iter, resi=resi,", src, fixed = TRUE)
eval(parse(text = src))
D <- read.csv("pwt_zx.csv"); X <- as.matrix(D[, 4:14]); y <- D$y; index1 <- D$iso
N <- length(unique(index1)); Ti <- length(y) / N
v <- as.matrix(X[1:Ti, 11] / norm(X[1:Ti, 11], "2"))
t0 <- proc.time()
fit <- RPC(X = X, y = y, v = v, factor1 = 2, factor2 = 1, index1 = index1, max_iter = max_iter)
el <- as.numeric((proc.time() - t0)[3])
S <- svd(fit$hat.Theta %*% t(fit$hat.B))
write.csv(data.frame(rss = fit$rss_hist), paste0(out, "_rss.csv"), row.names = FALSE)
write.csv(fit$hat.Theta %*% t(fit$hat.B), paste0(out, "_slopes.csv"), row.names = FALSE)
write.csv(fit$hat.F, paste0(out, "_F.csv"), row.names = FALSE); write.csv(fit$hat.Theta, paste0(out, "_theta.csv"), row.names = FALSE)
write.csv(data.frame(n_iter = fit$n_iter, time = el, RSS = fit$RSS, sv1 = S$d[1], sv2 = S$d[2]), paste0(out, "_info.csv"), row.names = FALSE)
write.csv(round(S$u[, 1:2], 3), paste0(out, "_U.csv"), row.names = FALSE)
