'use strict';

module.exports = (hexo) => {
  if (hexo.theme.has_hello) {
    return;
  }

  if (hexo.theme.i18n.languages[0].search(/zh-CN/i) !== -1) {
    hexo.log.info(`
114514
`);
  } else {
    hexo.log.info(`
114514
`);
  }

  hexo.theme.has_hello = true;
};
