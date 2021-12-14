BASEDIR=$(dirname "$0")
cd $BASEDIR/docs
mv html/* .
rm -r html

sed -i -e 's/_static/static/g' *.html
sed -i -e 's/_sources/sources/g' *.html
sed -i -e 's/_images/images/g' *.html
rm -rf static && mv _static static
rm -rf sources && mv _sources sources
rm -rf images && mv _images images
rm *.html-e
echo "Post sphinx done."
