
int main()
{
    int a;
    char c;
    a='a'-'A';
    scanf("%c",&c);
    if ('a'<=c<='z')
    {
        printf("%c",c-a);
    }
    else if ('A'<=c<='Z')
    {
}
    else {
        printf("%c",c);
    }
    
	return 0;
}