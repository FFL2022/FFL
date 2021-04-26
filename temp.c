int main()
{
    char ch;
    scanf("%c",&ch);
    if((ch>='a')&&(ch<='z'))
    {
        ch=ch-'a'+'A';
        printf("%c",ch);}
        if((ch>='A')&&(ch<='Z'))
        {
            ch=ch-'A'+'a';
        }
        else if((ch>='0')&&(ch<='9'))
        {
            int a= (int)ch-9;
            printf("%d",a);}
            else
            printf("%c",ch);
	return 0;
}