import torch
l1 = list()
a = torch.arange(-10,30)
a = a.view(2,2,5,2)
print(a)
a1 = a.view(4,5,2)
a2 = a1.view(2,2,5,2)
print(a1)
print(a2)





l1.append(a)
b = torch.arange(10,20)
b = b.view(1,2,5)
l1.append(b)
max_len_question_mask = max([i.size(1) for i in l1])
question_mask_list = list()
for i in l1:
    # i.cpu()
    if max_len_question_mask > i.size(1):
        d,e,f= i.size()
        padding = torch.full((d, max_len_question_mask-e, f), 0).long()
        new_ques_mask = torch.cat((i, padding), dim=1)
        question_mask_list.append(new_ques_mask)
    else:
        question_mask_list.append(i)

c = torch.cat(question_mask_list,dim=0)
# c = c.view(2,5,5)
# print(a)
# print(b)
# print('c.size(): ',c.size(),'\n','c: ',c)
#
#
#
# a=torch.arange(0,10)
# a = a.view(2,5)
# b = a.view(5,2)
# c = a.permute(1,0)
# print(a)
# print(b.size(),"\n",b)
# print(c.size(),"\n",c)