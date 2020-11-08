function [sales_vec] = count_sales(month, sale_cnt)
    sales_vec = zeros(1, 34);
    sales_vec(month + 1) = sale_cnt;
end